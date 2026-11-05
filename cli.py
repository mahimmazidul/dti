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
    parser.add_argument("--graph", action="store_true", help="Generate drug-target network graph")
    parser.add_argument("--save-graph", type=str, default=None, help="Filename to save graph PNG")
    parser.add_argument("--repurpose", action="store_true", help="Scan for repurposing candidates hitting this target")
    parser.add_argument("--min-phase", type=int, default=2, help="Minimum clinical phase for repurposing candidates (0-4)")
    
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

    # Network Graph section
    if args.graph:
        if args.mode == "drug":
            print("Network graph is only available in target mode")
        else:
            from src.visualizer import build_interaction_network, summarize_network
            
            # Get bioactivities from results
            bioactivities = []
            if results.get("results"):
                # Extract bioactivities from the results dict structure
                # In target mode, results["results"] is a list of dicts with molecule info
                # We need to reconstruct or use the original bioactivities from analyzer logic
                # Since analyzer.py doesn't return the raw bioactivities list directly, 
                # we'll reconstruct it or modify analyzer.py if needed.
                # Actually, let's look at analyzer.py - it returns results list with ic50_nm, pchembl, etc.
                # We can map them back to the format expected by visualizer.
                for item in results["results"]:
                    bioactivities.append({
                        "molecule_chembl_id": item.get("drug_name"), # Note: item["drug_name"] might be IUPAC name or ID
                        "standard_value": item.get("ic50_nm"),
                        "pchembl_value": item.get("pchembl"),
                        "canonical_smiles": item.get("smiles")
                    })
            
            # Print summary
            summary = summarize_network(args.query, bioactivities)
            import pandas as pd
            print("\n" + "-" * 50)
            print("Network Summary")
            print(pd.Series(summary).to_string())
            
            # Build network
            build_interaction_network(args.query, bioactivities, output_file=args.save_graph)

    # Repurposing Scanner section
    if args.repurpose:
        if args.mode == "drug":
            print("Repurposing scan is only available in target mode")
        else:
            from src.repurposing import find_repurposing_candidates, format_repurposing_results
            print("Scanning for repurposing candidates... this may take a moment")
            candidates = find_repurposing_candidates(args.query, min_phase=args.min_phase)
            print(format_repurposing_results(candidates))
            print("\nNote: computational repurposing candidates require experimental validation before any clinical conclusion can be drawn.")

    if __name__ == "__main__":
        main()
