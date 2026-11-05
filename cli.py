import argparse
import pandas as pd
from src.analyzer import analyze, format_results
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="DTI Analyzer - mahimmazidul")
    parser.add_argument("query", help="the drug name or target name to search")
    parser.add_argument("--mode", choices=["target", "drug"], default="target", help="search mode (default: target)")
    parser.add_argument("--limit", type=int, default=10, help="how many results to return (default: 10)")
    parser.add_argument("--save", help="optional filename to save results as CSV")
    parser.add_argument("--safety", action="store_true", help="Include FDA safety and interaction data")
    
    args = parser.parse_args()
    
    # Header
    print("DTI Analyzer - mahimmazidul")
    print("Data sources: ChEMBL, PubChem (free, open access)")
    print("-" * 50)
    
    # Run analysis
    results = analyze(args.query, mode=args.mode)
    
    # Print formatted results
    formatted = format_results(results, mode=args.mode)
    print(formatted)
    
    # Save to CSV if requested
    if args.save and results.get("results"):
        try:
            if args.mode == "target":
                # Convert target results to DataFrame
                df = pd.DataFrame(results["results"])
                # Select relevant columns
                df = df[["drug_name", "ic50_nm", "pchembl", "smiles"]]
                df.to_csv(args.save, index=False)
                print(f"\nResults saved to {args.save}")
            elif args.mode == "drug":
                # Convert drug results to DataFrame
                df = pd.DataFrame([results])
                df.to_csv(args.save, index=False)
                print(f"\nResults saved to {args.save}")
        except Exception as e:
            print(f"\nError saving to CSV: {e}")
            
    # FDA Safety Data section
    if args.safety and args.mode == "drug":
        time.sleep(1)
        print("\n" + "-" * 50)
        print("FDA Safety Data")
        
        fda_data = results.get("fda", {})
        interactions = fda_data.get("interactions_text", "No label data found")
        warnings = fda_data.get("warnings", "No warnings data found")
        events = fda_data.get("adverse_events", [])
        
        print(f"Drug Interactions: {interactions}")
        print(f"Warnings: {warnings}")
        
        if interactions == "No label data found":
            print(f"No FDA label data found for {args.query}")
        else:
            print("Top Adverse Events:")
            for i, event in enumerate(events, 1):
                print(f"  {i}. {event['reaction_name']} - Count: {event['count']}, Serious: {event['serious']}")

if __name__ == "__main__":
    main()
