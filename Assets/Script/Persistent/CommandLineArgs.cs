using System;
using System.Linq;
using UnityEngine;
using YARG.Core;
using YARG.Core.Game;

namespace YARG
{
    [DefaultExecutionOrder(-4999)]
    public static class CommandLineArgs
    {
        // Yes, the arguments should probably be prefixed with "--", however, this is based upon
        // Unity's existing command line arguments to make them consistent in style.

        /// <summary>
        /// Whether or not the game should be launched in offline mode. Offline mode disables
        /// online features such as fetching the OpenSource icons.
        /// </summary>
        private const string OFFLINE_ARG = "-offline";

        /// <summary>
        /// Defines whether we should save frame time data to replays
        /// </summary>
        private const string VERBOSE_REPLAYS = "-verbose-replays";

        /// <summary>
        /// Used to select the language the game will be launched in. The argument after should be
        /// the language code.
        /// </summary>
        private const string LANGUAGE_ARG = "-lang";

        /// <summary>
        /// Used to reference the download location of YARG and all of its setlists (by the launcher).
        /// The argument after should be the download location path.
        /// </summary>
        private const string DOWNLOAD_LOCATION_ARG = "-download-location";

        private const string PERSISTENT_DATA_PATH_ARG = "-persistent-data-path";

        // --- Automation (unattended queue playback) ---

        /// <summary>
        /// Path to a JSON queue describing the songs to play unattended. Presence of this
        /// argument is what switches the game into automation mode.
        /// </summary>
        private const string AUTO_QUEUE_ARG = "-autoqueue";

        /// <summary>
        /// Folder to scan for the queue's songs. Optional if the queue file sets "songDir".
        /// </summary>
        private const string AUTO_SONG_DIR_ARG = "-autosongdir";

        /// <summary>
        /// Folder for the automation outputs (score log, top scores, run summary).
        /// Defaults to &lt;persistent data&gt;/automation.
        /// </summary>
        private const string AUTO_DATA_DIR_ARG = "-autodata";

        /// <summary>
        /// Song speed for automated runs, as a percentage (100 = normal speed).
        /// </summary>
        private const string AUTO_SPEED_ARG = "-autospeed";

        private const string AUTO_INSTRUMENT_ARG = "-autoinstrument";
        private const string AUTO_DIFFICULTY_ARG = "-autodifficulty";

        /// <summary>
        /// Restart the queue from the beginning when it finishes instead of stopping.
        /// </summary>
        private const string AUTO_REPEAT_ARG = "-autorepeat";

        /// <summary>
        /// Keep the game running (returning to the menu) instead of quitting when the
        /// queue finishes. By default an automated run exits so a wrapper script can
        /// clean up.
        /// </summary>
        private const string AUTO_STAY_OPEN_ARG = "-autostayopen";

        public static bool Offline { get; private set; }

        public static bool VerboseReplays { get; private set; }

        public static string Language           { get; private set; }
        public static string DownloadLocation   { get; private set; }
        public static string PersistentDataPath { get; private set; }

        public static string AutoQueuePath     { get; private set; }
        public static string AutoSongDirectory { get; private set; }
        public static string AutoDataPath      { get; private set; }

        public static float AutoSongSpeed { get; private set; } = 1f;

        public static Instrument AutoInstrument { get; private set; } = Instrument.FiveFretGuitar;
        public static Difficulty AutoDifficulty { get; private set; } = Difficulty.Expert;

        public static bool AutoRepeatQueue   { get; private set; }
        public static bool AutoExitWhenDone  { get; private set; } = true;


        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSplashScreen)]
        private static void InitCommandLineArgs()
        {
            var args = Environment.GetCommandLineArgs();

            // Remember, the first argument is always the application itself
            for (int i = 1; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case OFFLINE_ARG:
                        Offline = true;
                        break;
                    case VERBOSE_REPLAYS:
                        VerboseReplays = true;
                        break;
                    case LANGUAGE_ARG:
                        i++;
                        if (i < args.Length)
                        {
                            Language = args[i];
                        }

                        break;
                    case DOWNLOAD_LOCATION_ARG:
                        i++;
                        if (i < args.Length)
                        {
                            DownloadLocation = args[i];
                        }

                        break;
                    case PERSISTENT_DATA_PATH_ARG:
                        i++;
                        if (i < args.Length)
                        {
                            PersistentDataPath = args[i];
                        }

                        break;

                    case AUTO_QUEUE_ARG:
                        i++;
                        if (i < args.Length)
                        {
                            AutoQueuePath = args[i];
                        }

                        break;
                    case AUTO_SONG_DIR_ARG:
                        i++;
                        if (i < args.Length)
                        {
                            AutoSongDirectory = args[i];
                        }

                        break;
                    case AUTO_DATA_DIR_ARG:
                        i++;
                        if (i < args.Length)
                        {
                            AutoDataPath = args[i];
                        }

                        break;
                    case AUTO_SPEED_ARG:
                        i++;
                        if (i < args.Length && float.TryParse(args[i], out var speedPercent))
                        {
                            // Accept both "150" and "150%".
                            AutoSongSpeed = Math.Clamp(speedPercent / 100f, 0.1f, 50f);
                        }

                        break;
                    case AUTO_INSTRUMENT_ARG:
                        i++;
                        if (i < args.Length && Enum.TryParse<Instrument>(args[i], true, out var instrument))
                        {
                            AutoInstrument = instrument;
                        }

                        break;
                    case AUTO_DIFFICULTY_ARG:
                        i++;
                        if (i < args.Length)
                        {
                            var difficulty = args[i].TrimEnd('%');
                            if (Enum.TryParse<Difficulty>(difficulty, true, out var parsedDifficulty))
                            {
                                AutoDifficulty = parsedDifficulty;
                            }
                        }

                        break;
                    case AUTO_REPEAT_ARG:
                        AutoRepeatQueue = true;
                        break;
                    case AUTO_STAY_OPEN_ARG:
                        AutoExitWhenDone = false;
                        break;
                }
            }
        }
    }
}