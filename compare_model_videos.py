import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
from src.inference import run_inference
from pathlib import Path

# Configure logging to show only INFO and above
logging.basicConfig(level=logging.INFO, format='%(message)s')
# Suppress some verbose loggers from libraries
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def compare():
    videos = ["sample videos/sample1.mp4", "sample videos/sample2.mp4"]
    
    # Ensure they exist
    for v in videos:
        if not os.path.exists(v):
            print(f"Video not found: {v}")
            return

    print("===========================================")
    print(" Running Model Comparison on Sample Videos ")
    print("===========================================")
    
    for video in videos:
        print(f"\nProcessing {video}...")
        out_name = Path(video).stem + "_comparison_output.json"
        
        try:
            result = run_inference(video, out_name, use_mudra_model=True)
            
            print(f"\n--- Results for {video} ---")
            mudra_summary = result.get('mudra_summary', {})
            if mudra_summary:
                print("Top Detected Mudras:")
                for mudra, count in mudra_summary.items():
                    print(f"  - {mudra}: {count} instance(s)")
            else:
                print("No mudras detected.")
                
            steps = result.get('dance_steps', [])
            print(f"Detected {len(steps)} dance segments.")
            
        except Exception as e:
            print(f"Error processing {video}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    compare()
