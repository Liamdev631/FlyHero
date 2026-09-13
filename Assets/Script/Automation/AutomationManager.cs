using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Cysharp.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using YARG.Core;
using YARG.Core.Game;
using YARG.Core.Logging;
using YARG.Core.Song;
using YARG.Gameplay;
using YARG.Gameplay.Player;
using YARG.Player;
using YARG.Settings;
using YARG.Song;

namespace YARG.Automation
{
    /// <summary>
    /// Runs YARG unattended: plays a queue of songs back to back, records the score of
    /// every attempt to the automation log, keeps a dictionary of the best score per
    /// song, and moves straight on to the next song when one ends.
    ///
    /// Without a player model attached, songs are played by a bot profile, which YARG
    /// hits every note with - so the loop produces real, non-zero scores.
    ///
    /// Enabled with the -autoqueue command line argument; see AutomationRunner script
    /// and AUTOMATION.md for the full workflow.
    /// </summary>
    public static class AutomationManager
    {
        private const string BotProfileName = "FlyHero AutoBot";

        private static readonly List<SongEntry> QueueSongs = new();
        private static readonly Stopwatch RunTimer = new();

        private static string _songDirectory;
        private static string _dataDirectory;

        private static bool _initialized;
        private static bool _finished;
        private static int  _attempts;

        public static bool   IsActive       => _initialized && !_finished;
        public static int    QueueIndex     { get; private set; }
        public static int    AttemptsLogged => _attempts;

        public static IReadOnlyList<SongEntry> Queue => QueueSongs;

        /// <summary>
        /// True when the game was launched with an automation queue configured.
        /// </summary>
        public static bool Requested => !string.IsNullOrWhiteSpace(CommandLineArgs.AutoQueuePath);

        /// <summary>
        /// Called from GlobalVariables.Start when automation was requested.
        /// Takes over the boot sequence; the caller should not load the menu.
        /// </summary>
        public static void Begin()
        {
            if (_initialized)
            {
                return;
            }

            _initialized = true;
            RunTimer.Start();

            _dataDirectory = CommandLineArgs.AutoDataPath;
            if (string.IsNullOrWhiteSpace(_dataDirectory))
            {
                // CommandLineArgs.PersistentDataPath, not PathHelper's: the
                // latter is not public and fails with CS0122.
                _dataDirectory = Path.Combine(CommandLineArgs.PersistentDataPath, "automation");
            }

            AutomationScoreStore.Initialize(_dataDirectory);
            AutomationObserver.Initialize(_dataDirectory);

            YargLogger.LogInfo("Automation mode starting up...");
            BeginAsync().Forget();
        }

        private static async UniTaskVoid BeginAsync()
        {
            try
            {
                QueueSongs.Clear();

                if (!LoadQueue())
                {
                    YargLogger.LogError("Automation queue could not be loaded; falling back to the menu.");
                    _finished = true;
                    GlobalVariables.Instance.LoadScene(SceneIndex.Menu);
                    return;
                }

                // Make the queue's song folder discoverable, then scan for songs.
                await ScanSongsAsync();

                if (!ResolveQueueSongs())
                {
                    _finished = true;
                    GlobalVariables.Instance.LoadScene(SceneIndex.Menu);
                    return;
                }

                RegisterBotPlayer();

                QueueIndex = 0;

                var first = QueueSongs[0];
                YargLogger.LogInfo($"Automation queue loaded: {QueueSongs.Count} song(s). Starting with \"{first.Name}\"");

                GlobalVariables.State = PersistentState.Default;
                GlobalVariables.State.PlayingAShow = true;
                GlobalVariables.State.ShowSongs = new List<SongEntry>(QueueSongs);
                GlobalVariables.State.ShowIndex = 0;
                GlobalVariables.State.CurrentSong = first;
                GlobalVariables.State.SongSpeed = CommandLineArgs.AutoSongSpeed;

                GlobalVariables.Instance.LoadScene(SceneIndex.Gameplay);
            }
            catch (Exception e)
            {
                YargLogger.LogException(e, "Automation failed to start.");
                _finished = true;
                GlobalVariables.Instance.LoadScene(SceneIndex.Menu);
            }
        }

        // ---------------------------------------------------------------- queue

        private static bool LoadQueue()
        {
            var queuePath = CommandLineArgs.AutoQueuePath;

            if (!File.Exists(queuePath))
            {
                YargLogger.LogError($"Automation queue file not found: {queuePath}");
                return false;
            }

            try
            {
                var json = File.ReadAllText(queuePath);
                var root = JToken.Parse(json);

                if (root is JArray array)
                {
                    // Bare array of song folder names or paths.
                    _songDirectory = CommandLineArgs.AutoSongDirectory;
                    QueueSongs.Clear();
                    _queueNames = array.Values<string>().Where(v => !string.IsNullOrWhiteSpace(v)).ToList();
                }
                else if (root is JObject obj)
                {
                    _songDirectory = obj.Value<string>("songDir") ?? CommandLineArgs.AutoSongDirectory;
                    var songs = obj["songs"]?.Values<string>() ?? Enumerable.Empty<string>();
                    _queueNames = songs.Where(v => !string.IsNullOrWhiteSpace(v)).ToList();
                }
                else
                {
                    YargLogger.LogError("Automation queue must be a JSON array of songs or an object with a \"songs\" array.");
                    return false;
                }

                if (_queueNames.Count == 0)
                {
                    YargLogger.LogError("Automation queue is empty.");
                    return false;
                }

                if (string.IsNullOrWhiteSpace(_songDirectory))
                {
                    // Fall back to the folder holding the queue file.
                    _songDirectory = Path.GetDirectoryName(Path.GetFullPath(queuePath));
                    YargLogger.LogWarning($"No songDir set; using the queue file's folder: {_songDirectory}");
                }

                // Interpolated: LogFormatInfo<T1>/<T2> are ambiguous for (int, string).
                YargLogger.LogInfo($"Automation queue: {_queueNames.Count} entries, songs in {_songDirectory}");
                return true;
            }
            catch (Exception e)
            {
                YargLogger.LogException(e, "Failed to parse the automation queue file.");
                return false;
            }
        }

        private static List<string> _queueNames = new();

        private static async UniTask ScanSongsAsync()
        {
            if (!string.IsNullOrWhiteSpace(_songDirectory) &&
                !SettingsManager.Settings.SongFolders.Contains(_songDirectory))
            {
                SettingsManager.Settings.SongFolders.Add(_songDirectory);
            }

            YargLogger.LogInfo("Automation scanning for songs...");
            // Full (non-quick) refresh so newly added folders are picked up.
            //
            // RunRefresh's context parameter is optional AND nullable, but
            // SongSources.LoadSprites dereferences it unconditionally
            // (context.SetLoadingText), so passing null throws
            // NullReferenceException. Normal startup always supplies one; our
            // path does not, so we create one here.
            try
            {
                using var context = new LoadingContext();
                await SongContainer.RunRefresh(false, context);
            }
            catch (Exception e)
            {
                // The scan itself usually succeeded (SongContainer is populated);
                // the failure is in cosmetic steps such as sprite loading. Do not
                // let that abort an unattended run.
                YargLogger.LogException(e, "Automation: song scan reported an error; continuing.");
            }

            YargLogger.LogInfo($"Automation found {SongContainer.Count} song(s) in the library");
        }

        /// <summary>
        /// Maps each queue entry onto a loaded song. Entries may be a song folder name,
        /// a "Name - Artist" string, or a song folder path.
        /// </summary>
        private static bool ResolveQueueSongs()
        {
            var library = SongContainer.Songs;
            var missing = new List<string>();

            foreach (var name in _queueNames)
            {
                var entry = FindSong(library, name);
                if (entry is null)
                {
                    missing.Add(name);
                    continue;
                }

                QueueSongs.Add(entry);
            }

            if (missing.Count > 0)
            {
                // Interpolated rather than LogFormatError: the generic
                // LogFormatError<T1>/<T2> overloads are ambiguous for an
                // (int, string) pair.
                YargLogger.LogError(
                    $"Automation could not find {missing.Count} song(s) in the library: {string.Join(", ", missing)}");
            }

            if (QueueSongs.Count == 0)
            {
                YargLogger.LogError("Automation resolved no songs; nothing to play.");
                return false;
            }

            return true;
        }

        private static SongEntry FindSong(SongEntry[] library, string name)
        {
            var trimmed = name.Trim();

            // 1. Exact folder path.
            var byPath = library.FirstOrDefault(s =>
                string.Equals(s.ActualLocation, trimmed, StringComparison.OrdinalIgnoreCase));
            if (byPath is not null)
            {
                return byPath;
            }

            // 2. Folder name.
            var byFolder = library.FirstOrDefault(s =>
                string.Equals(SafeFolderName(s), trimmed, StringComparison.OrdinalIgnoreCase));
            if (byFolder is not null)
            {
                return byFolder;
            }

            // 3. "Name - Artist".
            var byBoth = library.FirstOrDefault(s =>
                string.Equals($"{s.Name} - {s.Artist}", trimmed, StringComparison.OrdinalIgnoreCase));
            if (byBoth is not null)
            {
                return byBoth;
            }

            // 4. Song title.
            return library.FirstOrDefault(s =>
                string.Equals(s.Name, trimmed, StringComparison.OrdinalIgnoreCase));
        }

        private static string SafeFolderName(SongEntry song)
        {
            try
            {
                var location = song.ActualLocation;
                if (string.IsNullOrEmpty(location))
                {
                    return string.Empty;
                }

                return new DirectoryInfo(location).Name;
            }
            catch
            {
                return string.Empty;
            }
        }

        // --------------------------------------------------------------- player

        /// <summary>
        /// Ensures a bot profile exists and is active. Bots skip input processing in the
        /// engine and hit every note at its exact time, so the run needs no device and
        /// still produces a real score.
        /// </summary>
        private static void RegisterBotPlayer()
        {
            // Automation runs the bot alone. Profiles left over from earlier
            // sessions (e.g. a saved human profile) otherwise stay in the band,
            // contributing a zero score and making band-level aggregates - and
            // the all-players-FC flag - meaningless.
            var stale = PlayerContainer.Players.ToArray();
            if (stale.Length > 0)
            {
                foreach (var player in stale)
                {
                    PlayerContainer.DisposePlayer(player);
                }

                YargLogger.LogInfo($"Automation removed {stale.Length} pre-existing player(s) from the band");
            }

            var profile = PlayerContainer.Profiles.FirstOrDefault(p => p.IsBot && p.Name == BotProfileName);

            if (profile is null)
            {
                profile = new YargProfile
                {
                    Name = BotProfileName,
                    NoteSpeed = 5,
                    HighwayLength = 1,
                    GameMode = GameMode.FiveFretGuitar,
                    IsBot = true
                };

                if (!PlayerContainer.AddProfile(profile))
                {
                    YargLogger.LogError("Automation could not add the bot profile.");
                    return;
                }

                YargLogger.LogInfo($"Automation created bot profile \"{BotProfileName}\"");
            }

            profile.CurrentInstrument  = CommandLineArgs.AutoInstrument;
            profile.CurrentDifficulty  = CommandLineArgs.AutoDifficulty;

            if (!PlayerContainer.IsProfileTaken(profile))
            {
                // resolveDevices: false - a bot has no hardware bound to it.
                PlayerContainer.CreatePlayerFromProfile(profile, false);
            }
        }

        // ------------------------------------------------------------ song end

        /// <summary>
        /// Called by GameManager.EndSong when a song finishes. Logs the result and either
        /// starts the next song in the queue or ends the run.
        /// </summary>
        public static void OnSongFinished(GameManager gameManager)
        {
            if (!IsActive)
            {
                return;
            }

            try
            {
                var result = BuildResult(gameManager);
                AutomationScoreStore.Record(result);
                _attempts++;
            }
            catch (Exception e)
            {
                YargLogger.LogException(e, "Automation failed to record the song result.");
            }

            Advance();
        }

        private static AutomationSongResult BuildResult(GameManager gameManager)
        {
            var song = gameManager.Song;

            return new AutomationSongResult
            {
                SongKey           = SongKey(song),
                SongName          = song?.Name ?? "(unknown)",
                SongArtist        = song?.Artist ?? string.Empty,
                SongCharter       = song?.Charter ?? string.Empty,
                SongFolder        = song is null ? string.Empty : SafeFolderName(song),
                QueueIndex        = QueueIndex,
                AttemptNumber     = _attempts + 1,
                BandScore         = gameManager.BandScore,
                BandStars         = (int) gameManager.BandStars,
                SongSpeed         = gameManager.SongSpeed,
                SongLengthSeconds = gameManager.SongLength,
                Players           = gameManager.Players.Select(BuildPlayerResult).ToArray()
            };
        }

        private static AutomationPlayerResult BuildPlayerResult(BasePlayer player)
        {
            var stats = player.BaseStats;

            return new AutomationPlayerResult
            {
                Name        = player.Player.Profile.Name,
                Instrument  = player.Player.Profile.CurrentInstrument.ToString(),
                Difficulty  = player.Player.Profile.CurrentDifficulty.ToString(),
                IsBot       = player.Player.Profile.IsBot,
                Score       = player.Score,
                Stars       = (int) stats.Stars,
                NotesHit    = stats.NotesHit,
                NotesMissed = stats.NotesMissed,
                MaxCombo    = stats.MaxCombo,
                Percent     = stats.Percent,
                IsFc        = player.IsFc
            };
        }

        private static string SongKey(SongEntry song)
        {
            if (song is null)
            {
                return "unknown";
            }

            try
            {
                var bytes = song.Hash.HashBytes;
                if (bytes is { Length: > 0 })
                {
                    // Not Convert.ToHexString: that is .NET 5+, and Unity's
                    // Mono profile does not provide it.
                    return BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant();
                }
            }
            catch
            {
                // Fall through to a name based key.
            }

            var folder = SafeFolderName(song);
            return string.IsNullOrEmpty(folder) ? $"{song.Name} - {song.Artist}" : folder;
        }

        private static void Advance()
        {
            var next = QueueIndex + 1;

            if (next >= QueueSongs.Count)
            {
                if (CommandLineArgs.AutoRepeatQueue)
                {
                    YargLogger.LogInfo("Automation queue finished; -autorepeat is set so it is starting over.");
                    next = 0;
                }
                else
                {
                    Finish();
                    return;
                }
            }

            QueueIndex = next;
            var song = QueueSongs[next];

            YargLogger.LogInfo($"Automation advancing to queue index {next}: \"{song.Name}\"");

            GlobalVariables.State.PlayingAShow = true;
            GlobalVariables.State.ShowIndex = next;
            GlobalVariables.State.CurrentSong = song;

            GlobalVariables.Instance.LoadScene(SceneIndex.Gameplay);
        }

        private static void Finish()
        {
            _finished = true;
            RunTimer.Stop();

            YargLogger.LogInfo($"Automation run complete: {_attempts} attempt(s) logged across {QueueSongs.Count} song(s).");

            AutomationScoreStore.WriteRunSummary(QueueSongs.Count, _attempts, RunTimer.Elapsed);

            GlobalVariables.State = PersistentState.Default;

            if (CommandLineArgs.AutoExitWhenDone)
            {
                YargLogger.LogInfo("Automation exiting.");
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#else
                Application.Quit(0);
#endif
            }
            else
            {
                GlobalVariables.Instance.LoadScene(SceneIndex.Menu);
            }
        }
    }
}
