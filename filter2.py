import sys
import uproot
import awkward as ak
import numpy as np


## MODIFICARE A MANO E_I IN SELECTION PRIMA DI FILTRARE IL ROOT FILE!!!

def process_root_file(file_location):
    file = uproot.open(file_location)
    tree = file['Hits']
    branches = ["eventID", "processName", "PDGEncoding","rsectorID", "edep", "posX", "posY", "posZ", 
            "sourcePosX", "sourcePosY", "sourcePosZ"]
    data = tree.arrays(branches,library="ak")
    return data[ak.argsort(data["eventID"])]

def selection(raw_data):
    E_i = 511  # keV
    compt_shoulder = (2/(2 + 511/E_i) * E_i)/1000   # keV

    total_detected = len(np.unique(ak.to_numpy(raw_data['eventID'])))
    print(f"\nTotal raw events:     {raw_data['eventID'][-1]:.2e}, "
      f"Detected raw events:  {total_detected:.2e} ({(total_detected / (raw_data['eventID'][-1])) * 100:.2f}%) ")

    raw_data = raw_data[raw_data["PDGEncoding"] == 22]
    groups = ak.unflatten(raw_data, ak.run_lengths(raw_data["eventID"]))
    initial_events = len(groups)
    

    one_hit     = groups[ak.num(groups) == 1]
    lost_compton= one_hit[(one_hit['processName'][:, 0] == "compt")  &
                          (one_hit['edep'][:,0] < compt_shoulder) ]
    
    enough_hits = groups[ak.num(groups) >1]
    processName = enough_hits["processName"]
    edep        = enough_hits['edep']
    rsectorID   = enough_hits['rsectorID']

    total_compt_det = ak.sum(groups[:,0]['processName'] == "compt") 
    compt_over_shoulder = ak.sum( (groups['processName'][:, 0] == "compt")  & (groups['edep'][:,0] > compt_shoulder) )
    print(f"\nTotal Compton events: {total_compt_det:.2e}, "
      f"Edep > Compt.Shoulder: {compt_over_shoulder:.2e} ({(compt_over_shoulder/total_compt_det)*100:.2f}%), "
      f"Lost after 1 hit : {len(lost_compton['eventID']):.2e}({(len(lost_compton['eventID'])/total_compt_det)*100:.2f}%)\n")



    
    element_selection = (
    
        (processName[:, 0] == "compt")  &
        (edep[:,0] < compt_shoulder) &
        (rsectorID[:,0] == rsectorID[:,1]) &
        ( (processName[:, 1] == "compt") | (processName[:, 1] == "phot")) 
        #(processName[:, 1] == "compt")
        #(edep[:,0] > 0.05) & ((edep[:,1] > 0.05))                        
    )
    
    
    selected_groups = enough_hits[element_selection][:, :2]  
    #selected_groups = groups[element_selection][:, :1]   
    #    
    data = ak.flatten(selected_groups)
    data_lost = ak.flatten(lost_compton)

    final_events = len(selected_groups)
    percentage = (final_events / initial_events) * 100 if initial_events > 0 else 0

    metadata = {
        "totFirstCompt": np.array([ak.sum(groups[:,0]['processName'] == "compt")]),
        "n_rawEvents": np.array([initial_events]),
        "n_filtEvents": np.array([final_events]),
        "percentageKept": np.array([percentage]),
        'sourceEnergy': np.array([E_i])
    }

    
    return metadata, data, data_lost

def save_to_root(data,data_l, metadata, filename, tree_name="Hits", keys_to_save=None):

    if keys_to_save is None:
        keys_to_save = data.fields  # default: save all fields

    arrays = {key: ak.to_numpy(data[key]) for key in keys_to_save}
    arrays_l = {key: ak.to_numpy(data_l[key]) for key in keys_to_save}

    with uproot.recreate(filename) as f:
        f[tree_name] = arrays
        f["Metadata"] = metadata
        f["Lost_compt"] = arrays_l

def main():
    if len(sys.argv) != 3:
        print("Usage: python filter_root_hits.py <input_root_file> <output_root_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Step 1: Load and process
    raw_data = process_root_file(input_file)

    # Step 2: Selection
    metadata, selected_data, data_l = selection(raw_data)

    # Step 3: Choose which fields to save
    fields_to_save = [
        "edep","rsectorID",
        "posX", "posY", "posZ",
        "sourcePosX", "sourcePosY", "sourcePosZ",
        "eventID",
        "processName"
    ]

    # Step 4: Save to a new ROOT file
    save_to_root(selected_data, data_l, metadata, output_file, keys_to_save=fields_to_save)

    print(f"✅ Filtered ROOT file saved to: {output_file}")

if __name__ == "__main__":
    main()