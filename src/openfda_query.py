"""
An FDA drug label is the official prescribing information document that must accompany every approved drug sold in the United States. It contains all the required details about how to use the drug safely, what it treats, and what risks to watch for.

Adverse events are unwanted or harmful effects that happen when patients take a drug, like nausea, allergic reactions, or more serious problems. Tracking these events matters for drug safety because it helps doctors and regulators understand real-world risks that might not have been seen in initial clinical trials.

This data complements ChEMBL and PubChem because they tell different parts of the story. ChEMBL tells you what a drug binds to and how potently, PubChem tells you its chemical properties, and FDA tells you what actually happened to real patients using the drug in practice.
"""

import requests
import time

def get_drug_interactions(drug_name):
    """
    Searches the FDA drug label database for interaction information about a drug.
    First tries brand name search, then falls back to generic name search.
    Returns interaction text, warnings, and source information.
    """
    time.sleep(0.5)  # Respect rate limit
    
    # First attempt: brand name
    url = "https://api.fda.gov/drug/label.json"
    params = {
        "search": f'openfda.brand_name:"{drug_name}"',
        "limit": 1
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                result = results[0]
                interactions = result.get("drug_interactions", [""])[0]
                warnings = result.get("warnings_and_cautions", [""])[0]
                
                # Truncate if too long
                if len(interactions) > 1000:
                    interactions = interactions[:1000] + "..."
                if len(warnings) > 500:
                    warnings = warnings[:500] + "..."
                    
                return {
                    "drug_name": drug_name,
                    "interactions_text": interactions,
                    "warnings": warnings,
                    "source": "FDA Drug Label Database"
                }
    except Exception:
        pass
        
    # Second attempt: generic name
    time.sleep(0.5)
    params_generic = {
        "search": f'openfda.generic_name:"{drug_name}"',
        "limit": 1
    }
    
    try:
        response = requests.get(url, params=params_generic)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                result = results[0]
                interactions = result.get("drug_interactions", [""])[0]
                warnings = result.get("warnings_and_cautions", [""])[0]
                
                if len(interactions) > 1000:
                    interactions = interactions[:1000] + "..."
                if len(warnings) > 500:
                    warnings = warnings[:500] + "..."
                    
                return {
                    "drug_name": drug_name,
                    "interactions_text": interactions,
                    "warnings": warnings,
                    "source": "FDA Drug Label Database"
                }
    except Exception:
        pass
        
    # Nothing found
    return {
        "drug_name": drug_name,
        "interactions_text": "No label data found",
        "warnings": "No warnings data found",
        "source": "FDA Drug Label Database"
    }

def get_adverse_events(drug_name, limit=5):
    """
    Fetches adverse event reports from the FDA drug event database for a drug.
    Counts how often each reaction appears and whether any reports were serious.
    Returns a sorted list of reactions by frequency.
    """
    time.sleep(0.5)
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params)
        
        # Handle rate limit 429
        if response.status_code == 429:
            time.sleep(5)
            response = requests.get(url, params=params)
            
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            reaction_counts = {}
            reaction_serious = {}
            
            for result in results:
                patient = result.get("patient", {})
                reactions = patient.get("reaction", [])
                serious = result.get("serious", 2)  # Default to not serious
                
                for reaction in reactions:
                    reaction_name = reaction.get("reactionmeddrapt", "Unknown")
                    reaction_counts[reaction_name] = reaction_counts.get(reaction_name, 0) + 1
                    
                    # If any report for this reaction was serious, mark as Yes
                    if serious == 1:
                        reaction_serious[reaction_name] = "Yes"
                    elif reaction_name not in reaction_serious:
                        reaction_serious[reaction_name] = "No"
                        
            # Build sorted list
            events_list = []
            for name, count in reaction_counts.items():
                events_list.append({
                    "reaction_name": name,
                    "count": count,
                    "serious": reaction_serious.get(name, "No")
                })
                
            # Sort by count descending
            events_list.sort(key=lambda x: x["count"], reverse=True)
            return events_list
            
    except Exception:
        pass
        
    return []

def get_drug_label_summary(drug_name):
    """
    Combines drug interaction information and adverse event data into a single summary.
    Calls both interaction and adverse event functions and merges their results.
    Catches errors and includes them in the result instead of crashing.
    """
    summary = {
        "drug_name": drug_name,
        "interactions_text": "",
        "warnings": "",
        "adverse_events": [],
        "source": "FDA Drug Label Database"
    }
    
    try:
        interactions_data = get_drug_interactions(drug_name)
        summary["interactions_text"] = interactions_data.get("interactions_text", "")
        summary["warnings"] = interactions_data.get("warnings", "")
    except Exception as e:
        summary["error"] = f"Interactions error: {str(e)}"
        
    try:
        events_data = get_adverse_events(drug_name)
        summary["adverse_events"] = events_data
    except Exception as e:
        if "error" in summary:
            summary["error"] += f" | Events error: {str(e)}"
        else:
            summary["error"] = f"Events error: {str(e)}"
            
    return summary
