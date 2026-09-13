using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using YARG.Core.Logging;

namespace YARG.Automation
{
    /// <summary>
    /// One player's performance on a single attempt at a song.
    /// </summary>
    public sealed class AutomationPlayerResult
    {
        public string Name        { get; set; }
        public string Instrument  { get; set; }
        public string Difficulty  { get; set; }
        public bool   IsBot       { get; set; }
        public int    Score       { get; set; }
        public int    Stars       { get; set; }
        public int    NotesHit    { get; set; }
        public int    NotesMissed { get; set; }
        public int    MaxCombo    { get; set; }
        public float  Percent     { get; set; }
        public bool   IsFc        { get; set; }
    }

    /// <summary>
    /// The result of one song attempt. One of these is appended to the run log
    /// every time a song in the automation queue finishes.
    /// </summary>
    public sealed class AutomationSongResult
    {
        public string   Timestamp         { get; set; }
        public string   SongKey           { get; set; }
        public string   SongName          { get; set; }
        public string   SongArtist        { get; set; }
        public string   SongCharter       { get; set; }
        public string   SongFolder        { get; set; }
        public int      QueueIndex        { get; set; }
        public int      AttemptNumber     { get; set; }
        public int      BandScore         { get; set; }
        public int      BandStars         { get; set; }
        public float    SongSpeed         { get; set; }
        public double   SongLengthSeconds { get; set; }
        public bool     NewHighScore      { get; set; }
        public int      PreviousBestScore { get; set; }

        public AutomationPlayerResult[] Players { get; set; } = Array.Empty<AutomationPlayerResult>();
    }

    /// <summary>
    /// The best result ever recorded for a given song. This is the value stored in
    /// the top-score dictionary, one entry per song.
    /// </summary>
    public sealed class AutomationHighScore
    {
        public string SongName      { get; set; }
        public string SongArtist    { get; set; }
        public int    BestScore     { get; set; }
        public float  BestPercent   { get; set; }
        public int    BestStars     { get; set; }
        public int    BestMaxCombo  { get; set; }
        public bool   BestIsFc      { get; set; }
        public int    Plays         { get; set; }
        public string FirstPlayed   { get; set; }
        public string LastPlayed    { get; set; }
    }

    /// <summary>
    /// Writes the per-attempt run log and maintains the persistent dictionary of
    /// top scores (best score achieved per song).
    ///
    /// Outputs, all under the configured automation data directory:
    ///   score_log.jsonl   - append-only, one JSON object per finished song attempt
    ///   score_log.txt     - the same events as human-readable lines
    ///   top_scores.json   - dictionary of song key -> best result ever achieved
    /// </summary>
    public static class AutomationScoreStore
    {
        private static readonly JsonSerializerSettings SerializerSettings = new()
        {
            Formatting = Formatting.Indented,
            Culture = CultureInfo.InvariantCulture
        };

        private static readonly object Lock = new();

        private static string _jsonlPath;
        private static string _textPath;
        private static string _topScoresPath;

        private static Dictionary<string, AutomationHighScore> _topScores = new();

        public static string JsonlPath     => _jsonlPath;
        public static string TextPath      => _textPath;
        public static string TopScoresPath => _topScoresPath;

        public static IReadOnlyDictionary<string, AutomationHighScore> TopScores => _topScores;

        /// <summary>
        /// Points the store at a directory and loads any previously saved top scores.
        /// </summary>
        public static void Initialize(string directory)
        {
            lock (Lock)
            {
                Directory.CreateDirectory(directory);

                _jsonlPath     = Path.Combine(directory, "score_log.jsonl");
                _textPath      = Path.Combine(directory, "score_log.txt");
                _topScoresPath = Path.Combine(directory, "top_scores.json");

                LoadTopScores();

                YargLogger.LogFormatInfo("Automation score log: {0}", _jsonlPath);
                YargLogger.LogFormatInfo("Automation top scores: {0}", _topScoresPath);
            }
        }

        private static void LoadTopScores()
        {
            _topScores = new Dictionary<string, AutomationHighScore>();

            if (!File.Exists(_topScoresPath))
            {
                return;
            }

            try
            {
                var json = File.ReadAllText(_topScoresPath);
                if (string.IsNullOrWhiteSpace(json))
                {
                    return;
                }

                var loaded = JsonConvert.DeserializeObject<Dictionary<string, AutomationHighScore>>(json);
                if (loaded is not null)
                {
                    _topScores = loaded;
                }

                YargLogger.LogFormatInfo("Loaded {0} top score entries", _topScores.Count);
            }
            catch (Exception e)
            {
                YargLogger.LogException(e, "Failed to read automation top scores; starting a fresh dictionary.");
                _topScores = new Dictionary<string, AutomationHighScore>();
            }
        }

        /// <summary>
        /// Records a finished attempt: appends it to the log, folds it into the
        /// top-score dictionary, and persists both.
        /// </summary>
        public static void Record(AutomationSongResult result)
        {
            lock (Lock)
            {
                try
                {
                    var now = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
                    result.Timestamp ??= now;

                    // --- top score dictionary ---
                    _topScores.TryGetValue(result.SongKey, out var best);
                    if (best is null)
                    {
                        best = new AutomationHighScore
                        {
                            SongName   = result.SongName,
                            SongArtist = result.SongArtist,
                            FirstPlayed = now,
                            // Seed with a value below any real score so the first
                            // attempt always registers as an improvement.
                            BestScore  = int.MinValue
                        };
                        _topScores[result.SongKey] = best;
                    }

                    result.PreviousBestScore = best.BestScore == int.MinValue ? 0 : best.BestScore;
                    result.NewHighScore      = result.BandScore > result.PreviousBestScore;

                    best.Plays      = best.Plays + 1;
                    best.LastPlayed = now;

                    if (result.NewHighScore)
                    {
                        best.BestScore    = result.BandScore;
                        best.BestStars    = result.BandStars;
                        best.BestPercent  = MaxPlayerPercent(result);
                        best.BestMaxCombo = MaxPlayerCombo(result);
                        best.BestIsFc     = AllPlayersFc(result);
                    }

                    // First attempt on a song: the seeded negative value is meaningless
                    // to a reader, so normalise it now that a real attempt exists.
                    if (best.BestScore == int.MinValue)
                    {
                        best.BestScore = result.BandScore;
                        best.BestStars = result.BandStars;
                    }

                    // --- append-only logs ---
                    File.AppendAllText(_jsonlPath, JsonConvert.SerializeObject(result, Formatting.None) + Environment.NewLine);
                    File.AppendAllText(_textPath, BuildTextLine(result, best));

                    // --- persist dictionary ---
                    File.WriteAllText(_topScoresPath, JsonConvert.SerializeObject(_topScores, SerializerSettings));

                    YargLogger.LogFormatInfo("Automation recorded {0} score {1} (best {2})",
                        result.SongKey, result.BandScore, best.BestScore);
                }
                catch (Exception e)
                {
                    YargLogger.LogException(e, "Failed to write automation score record.");
                }
            }
        }

        private static float MaxPlayerPercent(AutomationSongResult result)
        {
            return result.Players.Length == 0 ? 0f : result.Players.Max(p => p.Percent);
        }

        private static int MaxPlayerCombo(AutomationSongResult result)
        {
            return result.Players.Length == 0 ? 0 : result.Players.Max(p => p.MaxCombo);
        }

        private static bool AllPlayersFc(AutomationSongResult result)
        {
            return result.Players.Length != 0 && result.Players.All(p => p.IsFc);
        }

        private static string BuildTextLine(AutomationSongResult result, AutomationHighScore best)
        {
            var sb = new StringBuilder();
            sb.Append('[').Append(result.Timestamp).Append("] ");
            sb.Append(result.SongName);
            if (!string.IsNullOrWhiteSpace(result.SongArtist))
            {
                sb.Append(" - ").Append(result.SongArtist);
            }

            sb.Append(" | score ").Append(result.BandScore);
            sb.Append(" | stars ").Append(result.BandStars);
            if (result.Players.Length > 0)
            {
                var p = result.Players.OrderByDescending(x => x.Score).First();
                sb.Append(" | ").Append(p.Instrument)
                  .Append(' ').Append(p.Difficulty)
                  .Append(" | ").Append(p.Percent.ToString("0.00", CultureInfo.InvariantCulture)).Append('%')
                  .Append(" | ").Append(p.NotesHit).Append(" hit / ").Append(p.NotesMissed).Append(" missed")
                  .Append(" | combo ").Append(p.MaxCombo);
                if (p.IsFc)
                {
                    sb.Append(" | FC");
                }
            }

            sb.Append(" | best ").Append(best.BestScore);
            sb.Append(" | play #").Append(best.Plays);
            if (result.NewHighScore)
            {
                sb.Append("  <-- NEW HIGH SCORE");
            }

            sb.AppendLine();
            return sb.ToString();
        }

        /// <summary>
        /// Writes an end-of-run summary next to the other automation outputs.
        /// </summary>
        public static void WriteRunSummary(int queuedSongs, int completedAttempts, TimeSpan elapsed)
        {
            lock (Lock)
            {
                try
                {
                    var directory = Path.GetDirectoryName(_topScoresPath);
                    if (directory is null)
                    {
                        return;
                    }

                    var sb = new StringBuilder();
                    sb.AppendLine("FlyHero automation run summary");
                    sb.AppendLine("==============================");
                    sb.AppendLine($"Finished:        {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                    sb.AppendLine($"Songs in queue:  {queuedSongs}");
                    sb.AppendLine($"Attempts logged: {completedAttempts}");
                    sb.AppendLine($"Elapsed:         {elapsed:hh\\:mm\\:ss}");
                    sb.AppendLine();
                    sb.AppendLine("Top scores (best per song)");
                    sb.AppendLine("--------------------------");

                    foreach (var (key, best) in _topScores.OrderByDescending(kv => kv.Value.BestScore))
                    {
                        sb.AppendLine($"{best.SongName,-40} {best.SongArtist,-24} {best.BestScore,10}  " +
                                      $"{best.BestStars} stars  {best.Plays} play(s)  {key}");
                    }

                    File.WriteAllText(Path.Combine(directory, "run_summary.txt"), sb.ToString());
                }
                catch (Exception e)
                {
                    YargLogger.LogException(e, "Failed to write automation run summary.");
                }
            }
        }
    }
}
