"""
Full end-to-end pipeline:

1. TCGA data pipeline
   → builds Data/gigatime/annotations.csv

2. GigaTIME + Survival pipeline
   → runs inference, feature extraction, Cox model, SHAP
"""

def run_tcga_pipeline():
    print("\n==============================")
    print("STEP 0: TCGA DATA PIPELINE")
    print("==============================")

    from tcga_pipeline.run_tcga_pipeline import main as tcga_main

    tcga_main()


def run_gigatime_pipeline():
    print("\n==============================")
    print("STEP 1: GigaTIME + SURVIVAL")
    print("==============================")

    from gigatime_analyzer.scripts.run_pipeline import main as giga_main

    giga_main()


def main():
    print("STARTING FULL END-TO-END PIPELINE")

    # Step 0: TCGA data
    run_tcga_pipeline()

    # Step 1+: ML pipeline
    run_gigatime_pipeline()

    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    main()