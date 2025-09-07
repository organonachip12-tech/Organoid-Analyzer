import os
import imagej
import jpype
import scyjava
from Config import DATA_DIR, IMAGES_FOLDER, FIJI_PATH, CASE_NAME, JAVA_ARGUMENTS
import pandas as pd

def track_with_trackmate(images_folder, case_name, fiji_path, include_spots_without_track_id = False, ignore_duplicate_warning = False, java_arguments = ""):

    # Check if output folder can be created
    output_folder_path = os.path.join(DATA_DIR, case_name)
    if os.path.exists(output_folder_path) and not ignore_duplicate_warning:
        user_input = ""

        while user_input.lower() != "y" or user_input.lower() != "n":
            user_input = input(f"\n\nA folder with case name: '{case_name}' already exists at: '{output_folder_path}' \n" +
                        f"Should the current run replace the previous results in the folder? (y/n): ")
    
            if user_input == "y":
                import shutil
                try:
                    shutil.rmtree(output_folder_path)
                    print(f"Folder '{output_folder_path}' and its contents deleted successfully.")
                except OSError as e:
                    print(f"Error: {e}")
                break
            elif user_input == "n":
                print("\nExiting current tracking run. Please change the case name.")
                return

    os.makedirs(output_folder_path, exist_ok=True)

    # -------------------------------------------------------------------
    # Launch Fiji with TrackMate
    # -------------------------------------------------------------------

    scyjava.config.add_options(java_arguments)
    ij = imagej.init(fiji_path, mode="interactive")
    print(ij.getApp().getInfo(True))

    # Import Java basic classes
    Model = jpype.JClass('fiji.plugin.trackmate.Model')
    TrackMate = jpype.JClass('fiji.plugin.trackmate.TrackMate')
    ThresholdDetectorFactory = jpype.JClass('fiji.plugin.trackmate.detection.ThresholdDetectorFactory')
    SimpleSparseLAPTrackerFactory = jpype.JClass('fiji.plugin.trackmate.tracking.jaqaman.SimpleSparseLAPTrackerFactory')
    TMUtils = jpype.JClass("fiji.plugin.trackmate.util.TMUtils")
    DetectionUtils = jpype.JClass("fiji.plugin.trackmate.detection.DetectionUtils")
    MaskUtils = jpype.JClass("fiji.plugin.trackmate.detection.MaskUtils")
    Integer = jpype.JClass("java.lang.Integer")


    for folder_path in sorted(os.listdir(images_folder)):
        folder_path = os.path.join(images_folder, folder_path)
        if not os.path.isdir(folder_path):
            continue  # skip files at root level
    
        all_tracks = []
        all_spots = []

        if os.path.exists(folder_path):
            print(f"Processing {folder_path} ...")
        else:
            print(f"FILEDNE {folder_path}")
            return
        
        # Load image
        FolderOpener = jpype.JClass("ij.plugin.FolderOpener")
        imp = FolderOpener.open(folder_path)

        # Convert Z slices to T slices
        HyperStackConverter = jpype.JClass("ij.plugin.HyperStackConverter")
        nframes = imp.getStackSize()
        imp = HyperStackConverter.toHyperStack(imp, 1, 1, nframes, "xyctz", "composite")    

        # Initialize Trackmate Model
        model = Model()
        settings = jpype.JClass("fiji.plugin.trackmate.Settings")(imp)

        # Calculate Threshold using Otsu
        channel, t = 0, 0 
        img = TMUtils.rawWraps(imp)
        im_frame = DetectionUtils.prepareFrameImg(img, channel, t)
        interval = TMUtils.getInterval(img, settings)  
        interval = DetectionUtils.squeeze(interval)
        threshold = MaskUtils.otsuThreshold(im_frame)

        

        print("Using auto threshold of: ", threshold)

        # Detector Settings
        settings.detectorFactory = ThresholdDetectorFactory()
        settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
        settings.detectorSettings = {
            'INTENSITY_THRESHOLD': threshold,  # auto threshold
            'TARGET_CHANNEL': Integer(1),
            'SIMPLIFY_CONTOURS': True,
        }
        
        # Tracker Settings
        settings.trackerFactory = SimpleSparseLAPTrackerFactory()
        settings.trackerSettings = settings.trackerFactory.getDefaultSettings() # almost good enough
        
        settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = True
        settings.trackerSettings['ALLOW_TRACK_MERGING'] = True
        settings.trackerSettings['LINKING_MAX_DISTANCE'] = 15.0
        settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 15.0
        settings.trackerSettings.put("MAX_FRAME_GAP", Integer(2))
        settings.tstart = 0
        settings.tend = imp.getNFrames() - 1
        settings.dt = 1.0

        # Get spot analyzer factories
        SpotFitEllipseAnalyzerFactory = jpype.JClass('fiji.plugin.trackmate.features.spot.SpotFitEllipseAnalyzerFactory')
        SpotIntensityMultiCAnalyzerFactory = jpype.JClass('fiji.plugin.trackmate.features.spot.SpotIntensityMultiCAnalyzerFactory')
        SpotShapeAnalyzerFactory = jpype.JClass('fiji.plugin.trackmate.features.spot.SpotShapeAnalyzerFactory')
        SpotContrastAndSNRAnalyzerFactory = jpype.JClass('fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzerFactory')

        settings.addSpotAnalyzerFactory(SpotFitEllipseAnalyzerFactory())
        settings.addSpotAnalyzerFactory(SpotIntensityMultiCAnalyzerFactory())
        settings.addSpotAnalyzerFactory(SpotShapeAnalyzerFactory())
        settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory())

        # Get track analyzers
        TrackBranchingAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackBranchingAnalyzer')
        TrackDurationAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackDurationAnalyzer')
        TrackIndexAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackIndexAnalyzer')
        TrackLocationAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackLocationAnalyzer')
        TrackMotilityAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackMotilityAnalyzer')
        TrackSpeedStatisticsAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackSpeedStatisticsAnalyzer')
        TrackSpotQualityFeatureAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackSpotQualityFeatureAnalyzer')
        
        settings.addTrackAnalyzer(TrackBranchingAnalyzer())
        settings.addTrackAnalyzer(TrackDurationAnalyzer())
        settings.addTrackAnalyzer(TrackIndexAnalyzer())
        settings.addTrackAnalyzer(TrackLocationAnalyzer())
        settings.addTrackAnalyzer(TrackSpeedStatisticsAnalyzer())
        settings.addTrackAnalyzer(TrackSpotQualityFeatureAnalyzer())
        settings.addTrackAnalyzer(TrackMotilityAnalyzer())

        # Run TrackMate and check if the inputs and processing run successfully
        trackmate = TrackMate(model, settings)

        if not trackmate.checkInput():
            print(str(trackmate.getErrorMessage()))
            print(f"TrackMate: Invalid input. {folder_path}")
            continue

        if not trackmate.process():
            print(str(trackmate.getErrorMessage()))
            print(f"TrackMate: Unable to process tracking. {folder_path}")
            continue

        # Feature models
        spot_collection = model.getSpots()
        feature_model = model.getFeatureModel()
        track_model = model.getTrackModel()

        # --- Track-level features ---
        for track_id in track_model.trackIDs(True):
            row = {
                "LABEL": f"Track_{track_id}",
            }

            for feature in [
                "TRACK_ID", "TRACK_INDEX", "NUMBER_SPOTS","NUMBER_GAPS","NUMBER_SPLITS","NUMBER_MERGES","NUMBER_COMPLEX",
                "LONGEST_GAP","TRACK_DURATION","TRACK_START","TRACK_STOP",
                "TRACK_DISPLACEMENT","TRACK_X_LOCATION","TRACK_Y_LOCATION","TRACK_Z_LOCATION",
                "TRACK_MEAN_SPEED","TRACK_MAX_SPEED","TRACK_MIN_SPEED","TRACK_MEDIAN_SPEED","TRACK_STD_SPEED",
                "TRACK_MEAN_QUALITY","TOTAL_DISTANCE_TRAVELED","MAX_DISTANCE_TRAVELED",
                "CONFINEMENT_RATIO","MEAN_STRAIGHT_LINE_SPEED","LINEARITY_OF_FORWARD_PROGRESSION",
                "MEAN_DIRECTIONAL_CHANGE_RATE"
            ]:
                val = feature_model.getTrackFeature(track_id, feature)
                row[feature] = float(val) if val is not None else None

            all_tracks.append(row)

        # --- Spot-level features ---
        for spot in spot_collection.iterable(True):
            track_id = model.getTrackModel().trackIDOf(spot)

            if include_spots_without_track_id or track_id is not None:
                row = {
                    "LABEL": spot.getName(),
                    "ID": spot.ID(),
                    "TRACK_ID": track_id,
                }

                for feature in [
                    "QUALITY","POSITION_X","POSITION_Y","POSITION_Z","POSITION_T","FRAME",
                    "RADIUS","VISIBILITY","MANUAL_SPOT_COLOR",
                    "MEAN_INTENSITY_CH1","MEDIAN_INTENSITY_CH1","MIN_INTENSITY_CH1","MAX_INTENSITY_CH1",
                    "TOTAL_INTENSITY_CH1","STD_INTENSITY_CH1","CONTRAST_CH1","SNR_CH1",
                    "ELLIPSE_X0","ELLIPSE_Y0","ELLIPSE_MAJOR","ELLIPSE_MINOR","ELLIPSE_THETA",
                    "ELLIPSE_ASPECTRATIO","AREA","PERIMETER","CIRCULARITY","SOLIDITY","SHAPE_INDEX"
                ]:
                    val = spot.getFeature(feature)
                    row[feature] = float(val) if val is not None else None
                all_spots.append(row)

        # Save results
        df_tracks = pd.DataFrame(all_tracks)
        df_spots = pd.DataFrame(all_spots)

        subcase_name = "_".join(folder_path.split(os.sep)[-2:])
        df_tracks.to_csv(os.path.join(output_folder_path, subcase_name + "_tracks.csv"), index=False)
        df_spots.to_csv(os.path.join(output_folder_path, subcase_name +"_spots.csv"), index=False)

        print("Done! Exported all track and spot features to CSV ")

if __name__ == "__main__":
    for subfolder in sorted(os.listdir(IMAGES_FOLDER)):
        images_subfolder = os.path.join(IMAGES_FOLDER, subfolder)
        track_with_trackmate(images_subfolder, 
                             CASE_NAME, 
                             FIJI_PATH, 
                             java_arguments = JAVA_ARGUMENTS,
                             include_spots_without_track_id = False, 
                             ignore_duplicate_warning = True, )