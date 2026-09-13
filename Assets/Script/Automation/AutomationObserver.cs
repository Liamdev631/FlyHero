using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using YARG.Core.Chart;
using YARG.Core.Logging;
using YARG.Gameplay;
using YARG.Gameplay.Player;

namespace YARG.Automation
{
    /// <summary>
    /// Emits the game's own state, once per sampled frame, to observations.jsonl.
    ///
    /// Why this exists: everything downstream needs a clock that only the game has.
    ///
    ///  * Pairing a recorded video frame with the notes that were on screen needs the
    ///    song time at that frame. Deriving it from the recording's audio does not work
    ///    (see todo.md) - the game's mix does not contain the song distinctly enough to
    ///    correlate, and our synthetic songs have a near-constant loudness envelope. The
    ///    game knows this number exactly; it should just say it.
    ///  * The reward function currently scores timing from a mean note offset, because
    ///    per-note offsets are not logged. A real clock makes per-note credit possible.
    ///  * A learned policy needs the same clock as an input.
    ///
    /// What it deliberately does NOT do yet: emit the upcoming-note horizon. That needs the
    /// note track's time model, and guessing at it would be worse than leaving it out. The
    /// clock and player state are the pieces that unblock work today.
    /// </summary>
    public static class AutomationObserver
    {
        /// <summary>Notes to look ahead when emitting the horizon.</summary>
        private const int HORIZON_NOTES = 16;

        /// <summary>
        /// How far back to start scanning. Notes are sorted by Time, so we binary-search to
        /// approximately here and then walk forward; the margin has to exceed the longest
        /// plausible sustain so a still-ringing long note is not skipped.
        /// </summary>
        private const double HORIZON_LOOKBACK_SECONDS = 5.0;

        /// <summary>Sample rate. 20 Hz is comfortably finer than the note windows.</summary>
        private const double SAMPLE_INTERVAL_SECONDS = 0.05;

        /// <summary>
        /// UTF-8 WITHOUT a byte-order mark. Encoding.UTF8 writes a BOM on the first
        /// write, and a BOM at the head of a .jsonl makes strict JSON parsers fail on
        /// line 1 - it broke the reader once already.
        /// </summary>
        private static readonly UTF8Encoding Utf8NoBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

        private static readonly Stopwatch WallClock = Stopwatch.StartNew();
        private static readonly object Lock = new();

        private static string _path;
        private static double _nextSampleTime;

        public static bool Enabled => !string.IsNullOrEmpty(_path);
        public static string Path_ => _path;

        public static int LinesWritten { get; private set; }

        /// <summary>Point the observer at the automation data directory.</summary>
        public static void Initialize(string directory)
        {
            if (string.IsNullOrWhiteSpace(directory))
            {
                return;
            }

            Directory.CreateDirectory(directory);
            _path = System.IO.Path.Combine(directory, "observations.jsonl");
            _nextSampleTime = double.NegativeInfinity;
            LinesWritten = 0;

            YargLogger.LogFormatInfo("Automation observations: {0}", _path);
        }

        /// <summary>
        /// Record one sample if the interval has elapsed. Called every frame from
        /// GameManager.Update, so it must stay cheap.
        /// </summary>
        public static void Sample(GameManager gameManager)
        {
            if (!Enabled || gameManager is null)
            {
                return;
            }

            double songTime;
            try
            {
                songTime = gameManager.SongTime;
            }
            catch (Exception)
            {
                // Song time is not valid before the song actually starts.
                return;
            }

            if (songTime < _nextSampleTime)
            {
                return;
            }

            _nextSampleTime = songTime + SAMPLE_INTERVAL_SECONDS;

            try
            {
                Write(gameManager, songTime);
            }
            catch (Exception e)
            {
                YargLogger.LogException(e, "Automation failed to write an observation sample.");
            }
        }

        private static void Write(GameManager gameManager, double songTime)
        {
            var players = new List<object>();

            foreach (var player in gameManager.Players)
            {
                if (player is null)
                {
                    continue;
                }

                var profile = player.Player?.Profile;
                int totalNotes = player.TotalNotes;

                players.Add(new
                {
                    name = profile?.Name ?? "unknown",
                    instrument = profile?.CurrentInstrument.ToString() ?? "unknown",
                    difficulty = profile?.CurrentDifficulty.ToString() ?? "unknown",
                    is_bot = profile?.IsBot ?? false,
                    score = player.Score,
                    combo = player.Combo,
                    notes_hit = player.NotesHit,
                    total_notes = totalNotes,
                    // Not the game's own percentage - a plain hit fraction, named honestly.
                    notes_hit_fraction = totalNotes > 0 ? (double)player.NotesHit / totalNotes : 0.0,
                    stars = player.Stars,
                    is_fc = player.IsFc,
                });
            }

            var horizon = BuildHorizon(gameManager, songTime);

            var sample = new
            {
                // Absolute wall time, so a screen recording started at a known moment can
                // be mapped onto song_time exactly. wall_ms is only relative to this
                // process, which is not enough to line up with an external recorder.
                unix_ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                wall_ms = WallClock.Elapsed.TotalMilliseconds,
                song_time = songTime,
                song_length = gameManager.SongLength,
                players,
                horizon,
            };

            string line = JsonConvert.SerializeObject(sample, Formatting.None);

            lock (Lock)
            {
                File.AppendAllText(_path, line + Environment.NewLine, Utf8NoBom);
                LinesWritten++;
            }
        }

        /// <summary>
        /// Upcoming guitar notes: per note the lane, seconds until it must be hit, its sustain
        /// length, and whether it is part of a chord. This is the observation a policy needs to
        /// decide *when* to act.
        ///
        /// WARNING: `lane` is GuitarNote.Fret, which is 1-BASED (green..orange = 1..5), while
        /// the chart and the rest of these tooling use 0-based lanes (0..4). The off-by-one is
        /// silent, so consumers must subtract 1.
        ///
        /// Guitar only for now. Other instruments have their own note types (DrumNote,
        /// GuitarNote for keys, vocals) and need their own branch; TrackPlayer.NoteTrack lives
        /// on the generic subclass, so there is no common accessor to reuse.
        /// </summary>
        private static List<object> BuildHorizon(GameManager gameManager, double songTime)
        {
            var horizon = new List<object>();

            FiveFretGuitarPlayer guitar = null;
            foreach (var player in gameManager.Players)
            {
                if (player is FiveFretGuitarPlayer found)
                {
                    guitar = found;
                    break;
                }
            }

            var notes = guitar?.NoteTrack?.Notes;
            if (notes is null || notes.Count == 0)
            {
                return horizon;
            }

            int start = LowerBound(notes, songTime - HORIZON_LOOKBACK_SECONDS);
            for (int i = start; i < notes.Count && horizon.Count < HORIZON_NOTES; i++)
            {
                var note = notes[i];
                if (note.TimeEnd < songTime)
                {
                    // Entirely in the past, including its sustain.
                    continue;
                }

                horizon.Add(new
                {
                    lane = note.Fret,
                    dt = note.Time - songTime,
                    sustain = note.TimeLength,
                    chord = note.IsChord,
                });
            }

            return horizon;
        }

        /// <summary>First index whose Time is &gt;= value (notes are sorted by Time).</summary>
        private static int LowerBound(IReadOnlyList<GuitarNote> notes, double value)
        {
            int lo = 0, hi = notes.Count;
            while (lo < hi)
            {
                int mid = lo + (hi - lo) / 2;
                if (notes[mid].Time < value)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid;
                }
            }
            return lo;
        }

        /// <summary>
        /// One-line summary for the run log, so a run states whether it produced
        /// observations and where they went.
        /// </summary>
        public static string Describe()
        {
            return Enabled
                ? $"Automation observations: {LinesWritten} sample(s) -> {_path}"
                : "Automation observations: disabled";
        }
    }
}
