import os
import pandas as pd

# ================================================
# FIXED JVM + ImageJ INITIALIZATION (Global Setup)
# ================================================

# MUST set JAVA_HOME before importing jpype or launching ImageJ
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home"

import scyjava
import imagej

# Required open modules for Java 17 + JPype
scyjava.config.add_options("--add-opens=java.base/java.lang=ALL-UNNAMED")
scyjava.config.add_options("--add-opens=java.base/java.util=ALL-UNNAMED")

# Memory for TrackMate
scyjava.config.add_options("-Xmx24g")

# Start ImageJ ONCE — starting twice will crash JPype
ij = imagej.init("/Applications/Fiji", mode="headless")
print("✔ ImageJ initialized:", ij.getApp().getInfo(True))

# JPype MUST be imported AFTER JVM config is done
import jpype

# Import project settings
from config import (
    DATA_DIR,
    FIJI_PATH,
    JAVA_ARGUMENTS,
    CELL_TRACKING_DATASET_CONFIGS,
)

from pathlib import Path


# =====================================================
# TRACKMATE FUNCTION
# =====================================================

def track_with_trackmate(
    images_folder,
    subcase_names,
    case_name,
    fiji_path,
    prefix="",
    specific_thresholds={},
    include_spots_without_track_id=False,
    ignore_duplicate_warning=False,
    java_arguments=""
):

    # --------------------------
    # OUTPUT FOLDER HANDLING
    # --------------------------
    output_folder_path = os.path.join(DATA_DIR, case_name)

    if os.path.exists(output_folder_path) and not ignore_duplicate_warning:
        user_input = ""
        while user_input.lower() not in ["y", "n"]:
            user_input = input(
                f"\nFolder '{case_name}' already exists at '{output_folder_path}'\n"
                "Replace it? (y/n): "
            )
        if user_input == "y":
            import shutil
            shutil.rmtree(output_folder_path)
            print(f"✔ Deleted existing folder: {output_folder_path}")
        else:
            print("Tracking cancelled. Rename case_name.")
            return

    os.makedirs(output_folder_path, exist_ok=True)

    # --------------------------
    # Use the global ImageJ
    # --------------------------
    global ij

    # ================================
    # Load required Java classes
    # ================================
    Model = jpype.JClass("fiji.plugin.trackmate.Model")
    TrackMate = jpype.JClass("fiji.plugin.trackmate.TrackMate")
    SimpleSparseLAPTrackerFactory = jpype.JClass(
        "fiji.plugin.trackmate.tracking.jaqaman.SimpleSparseLAPTrackerFactory"
    )
    TMUtils = jpype.JClass("fiji.plugin.trackmate.util.TMUtils")
    Integer = jpype.JClass("java.lang.Integer")
    Double = jpype.JClass("java.lang.Double")

    # Detector
    LogDetectorFactory = jpype.JClass("fiji.plugin.trackmate.detection.LogDetectorFactory")

    # Spot analyzers
    SpotFitEllipseAnalyzerFactory = jpype.JClass(
        "fiji.plugin.trackmate.features.spot.SpotFitEllipseAnalyzerFactory"
    )
    SpotIntensityMultiCAnalyzerFactory = jpype.JClass(
        "fiji.plugin.trackmate.features.spot.SpotIntensityMultiCAnalyzerFactory"
    )
    SpotShapeAnalyzerFactory = jpype.JClass(
        "fiji.plugin.trackmate.features.spot.SpotShapeAnalyzerFactory"
    )
    SpotContrastAndSNRAnalyzerFactory = jpype.JClass(
        "fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzerFactory"
    )

    # Track analyzers
    TrackBranchingAnalyzer = jpype.JClass(
        "fiji.plugin.trackmate.features.track.TrackBranchingAnalyzer"
    )
    TrackDurationAnalyzer = jpype.JClass(
        "fiji.plugin.trackmate.features.track.TrackDurationAnalyzer"
    )
    TrackIndexAnalyzer = jpype.JClass(
        "fiji.plugin.trackmate.features.track.TrackIndexAnalyzer"
    )
    TrackLocationAnalyzer = jpype.JClass(
        "fiji.plugin.trackmate.features.track.TrackLocationAnalyzer"
    )
    TrackMotilityAnalyzer = jpype.JClass(
        "fiji.plugin.trackmate.features.track.TrackMotilityAnalyzer"
    )
    TrackSpeedStatisticsAnalyzer = jpype.JClass(
        "fiji.plugin.trackmate.features.track.TrackSpeedStatisticsAnalyzer"
    )
    TrackSpotQualityFeatureAnalyzer = jpype.JClass(
        "fiji.plugin.trackmate.features.track.TrackSpotQualityFeatureAnalyzer"
    )
    DirectionalChangeAnalyzer = jpype.JClass(
        "fiji.plugin.trackmate.features.edges.DirectionalChangeAnalyzer"
    )

    # TIFF / ImageJ classes
    TiffDecoder = jpype.JClass("ij.io.TiffDecoder")
    FileInfo = jpype.JClass("ij.io.FileInfo")
    FileInfoVirtualStack = jpype.JClass("ij.plugin.FileInfoVirtualStack")
    ImagePlus = jpype.JClass("ij.ImagePlus")
    HyperStackConverter = jpype.JClass("ij.plugin.HyperStackConverter")
    JFileInfoArray = jpype.JArray(FileInfo)

    # Determine Round (Round1, Round2, …)
    round_name = os.path.basename(images_folder)

    # =====================================
    # PROCESS EACH STROMA FOLDER
    # =====================================
    for stroma_folder in sorted(os.listdir(images_folder)):

        full_path = os.path.join(images_folder, stroma_folder)
        if stroma_folder.startswith(".") or not os.path.isdir(full_path):
            continue

        print("\n==============================")
        print(f"Processing folder: {stroma_folder}")
        print("==============================")

        # Collect all .tif files
        all_tifs = sorted(Path(full_path).glob("*.tif"))
        if not all_tifs:
            print(f"⚠ No .tif files found in: {full_path}, skipping.")
            continue

        # Extract XY labels (XY1, XY2…)
        xy_labels = sorted({p.stem.split("_")[-1] for p in all_tifs})
        print("Detected XY positions:", xy_labels)

        # Determine stroma label
        if "Stroma_7" in stroma_folder or "_Stroma_7" in stroma_folder:
            stroma_label = "Stroma7"
        elif "Stroma_8" in stroma_folder or "_Stroma_8" in stroma_folder:
            stroma_label = "Stroma8"
        else:
            stroma_label = "UnknownStroma"

        # ----------------------------------------------------
        # TRACK EACH XY POSITION
        # ----------------------------------------------------
        for xy in xy_labels:
            print(f"\n➡ Processing XY position: {xy}")

            tif_files = sorted([str(p) for p in all_tifs if p.stem.endswith(xy)])
            if not tif_files:
                print(f" ⚠ No frames for {xy}, skipping.")
                continue

            print(f" Number of frames for {xy}: {len(tif_files)}")

            # --- Build virtual stack ---
            fi_array = JFileInfoArray(len(tif_files))

            for i, f in enumerate(tif_files):
                p = Path(f)
                decoder = TiffDecoder(str(p.parent) + "/", p.name)
                info_list = decoder.getTiffInfo()

                if not info_list:
                    raise RuntimeError(f"❌ Could not decode TIFF header for: {f}")

                fi_array[i] = info_list[0]

            stack = FileInfoVirtualStack(fi_array)
            imp = ImagePlus(f"{stroma_folder}_{xy}", stack)

            width, height, nframes = (
                imp.getWidth(),
                imp.getHeight(),
                imp.getStackSize(),
            )
            print(f" Image size {width} x {height}, frames = {nframes}")

            # --- Calibration ---
            cal = imp.getCalibration()
            cal.pixelWidth = 0.33
            cal.pixelHeight = 0.33
            cal.setUnit("µm")
            cal.setTimeUnit("frames")
            imp.setCalibration(cal)

            # Convert Z -> T
            imp = HyperStackConverter.toHyperStack(
                imp, 1, 1, nframes, "xyctz", "composite"
            )

            # -------------------
            # TrackMate Settings
            # -------------------
            model = Model()
            settings = jpype.JClass("fiji.plugin.trackmate.Settings")(imp)

            # Intensity threshold
            intensity_threshold = 2.0
            for key, val in specific_thresholds.items():
                if key in stroma_folder or key in xy:
                    intensity_threshold = float(val)

            print("Threshold:", intensity_threshold)

            # Detector settings
            settings.detectorFactory = LogDetectorFactory()
            settings.detectorSettings = {
                "DO_SUBPIXEL_LOCALIZATION": True,
                "RADIUS": 6.0,
                "THRESHOLD": Double(intensity_threshold),
                "TARGET_CHANNEL": Integer(0),
                "DO_MEDIAN_FILTERING": True,
            }

            # Tracker settings
            settings.trackerFactory = SimpleSparseLAPTrackerFactory()
            settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
            settings.trackerSettings["ALLOW_TRACK_SPLITTING"] = False
            settings.trackerSettings["ALLOW_TRACK_MERGING"] = False
            settings.trackerSettings["LINKING_MAX_DISTANCE"] = 25.0
            settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = 25.0
            settings.trackerSettings.put("MAX_FRAME_GAP", Integer(5))

            # Add analyzers
            settings.addSpotAnalyzerFactory(SpotFitEllipseAnalyzerFactory())
            settings.addSpotAnalyzerFactory(SpotIntensityMultiCAnalyzerFactory())
            settings.addSpotAnalyzerFactory(SpotShapeAnalyzerFactory())
            settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory())

            settings.addTrackAnalyzer(TrackBranchingAnalyzer())
            settings.addTrackAnalyzer(TrackDurationAnalyzer())
            settings.addTrackAnalyzer(TrackIndexAnalyzer())
            settings.addTrackAnalyzer(TrackLocationAnalyzer())
            settings.addTrackAnalyzer(TrackSpeedStatisticsAnalyzer())
            settings.addTrackAnalyzer(TrackSpotQualityFeatureAnalyzer())
            settings.addTrackAnalyzer(TrackMotilityAnalyzer())

            settings.addEdgeAnalyzer(DirectionalChangeAnalyzer())

            # Run TrackMate
            trackmate = TrackMate(model, settings)

            if not trackmate.checkInput():
                print("❌ Input error:", trackmate.getErrorMessage())
                continue

            print("Starting detection process using 10 threads.")
            if not trackmate.process():
                print("❌ TrackMate process error:", trackmate.getErrorMessage())
                continue

            # Retrieve results
            spot_collection = model.getSpots()
            feature_model = model.getFeatureModel()
            track_model = model.getTrackModel()

            print("# of tracks:", track_model.nTracks(True))
            print("Track IDs:", list(track_model.trackIDs(True)))

            # ----------------------------------------------------
            # Collect TRACK features
            # ----------------------------------------------------
            track_feature_keys = list(feature_model.getTrackFeatures())
            spot_feature_keys = list(feature_model.getSpotFeatures())

            all_tracks = []
            for track_id in track_model.trackIDs(True):
                row = {"LABEL": f"Track_{track_id}", "TRACK_ID": int(track_id)}
                for feature in track_feature_keys:
                    val = feature_model.getTrackFeature(track_id, feature)
                    row[feature] = float(val) if val is not None else None
                all_tracks.append(row)

            # ----------------------------------------------------
            # Collect SPOT features
            # ----------------------------------------------------
            all_spots = []
            for spot in spot_collection.iterable(True):
                track_id = track_model.trackIDOf(spot)
                if include_spots_without_track_id or track_id is not None:
                    row = {
                        "LABEL": spot.getName(),
                        "ID": int(spot.ID()),
                        "TRACK_ID": int(track_id) if track_id is not None else None,
                    }
                    for feature in spot_feature_keys:
                        val = spot.getFeature(feature)
                        row[feature] = float(val) if val is not None else None
                    all_spots.append(row)

            # Export to CSV
            df_spots = pd.DataFrame(all_spots).sort_values(by="TRACK_ID")
            df_tracks = pd.DataFrame(all_tracks).sort_values(by="TRACK_ID")

            subcase_name = f"{case_name}_{round_name}_{stroma_label}"

            spots_output_path = os.path.join(
                output_folder_path,
                f"{prefix}{subcase_name}_{xy}_spots.csv",
            )
            tracks_output_path = os.path.join(
                output_folder_path,
                f"{prefix}{subcase_name}_{xy}_tracks.csv",
            )

            df_spots.to_csv(spots_output_path, index=False)
            df_tracks.to_csv(tracks_output_path, index=False)

            print(f"✔ Exported spots → {spots_output_path}")
            print(f"✔ Exported tracks → {tracks_output_path}")


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    for case in CELL_TRACKING_DATASET_CONFIGS:

        config = CELL_TRACKING_DATASET_CONFIGS[case]
        images_root = config["images_folder"]  # ./Data/Raw

        for subfolder in sorted(os.listdir(images_root)):
            if subfolder.startswith("."):
                continue

            images_subfolder = os.path.join(images_root, subfolder)
            if not os.path.isdir(images_subfolder):
                continue

            track_with_trackmate(
                images_subfolder,
                config["subcase_names"],
                config["case_name"],
                FIJI_PATH,
                prefix=config["prefix"],
                specific_thresholds=config.get("specific_thresholds", {}),
                include_spots_without_track_id=False,
                ignore_duplicate_warning=True,
            )

    print("--------------------------")
    print("✔ Cell Tracking Complete!")
    print("--------------------------")
