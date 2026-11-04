# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def value_mapping(value, subspace):
    """Map the index value in a POSCAR filename to the actual physical property value"""
    if subspace == "bm_ternary_fcc":
        value_mapped = (value + 2.60)/0.16 
    elif subspace == "bm_quinary_fcc":
        value_mapped = (value + 9.04)/0.25
    elif subspace == "bm_quinary_bcc":
        value_mapped = (value + 2.38)/0.13
    elif subspace == "ltc_quinary_fcc":
        value_mapped = (value + 7.75)/5.88
    elif subspace == "ltc_quinary_bcc":
        value_mapped = (value + 5.03)/1.28
    return value_mapped

# Define the preset element type dictionaries for FCC and BCC
PHASE_ELEMENTS = {
    "bcc": ['Zr', 'V', 'Ti', 'Nb', 'Mo', 'Ru', 'Rh', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au'],
    "fcc": ['Ti', 'Cr', 'W', 'Re', 'Ta', 'Ru', 'Rh', 'Os', 'Ir', 'Pd', 'Pt']
}

# Define Y-axis ranges for each physical property
Y_AXIS_RANGES = {
    "element_ratio": (0, 0.8),          # element ratio (fraction)
    "bm": (240, 280),                 # bulk modulus
    "ltc_bcc": (15, 35),              # BCC lattice thermal conductivity
    "ltc_fcc": (5, 15)                # FCC lattice thermal conductivity
}

def parse_poscar(poscar_path, phase):
    """
    Parse a POSCAR file and return the atomic-number ratios (fractions, 0-1) of the preset elements for the given phase
    :param poscar_path: path to the POSCAR file
    :param phase: phase type (fcc/bcc)
    :return: dictionary of element ratios
    """
    element_ratio = {}
    # Get the preset element list for the given phase
    required_elements = PHASE_ELEMENTS[phase]
    
    try:
        with open(poscar_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Parse element species and atom counts (compatible with line offsets of different POSCAR formats)
        elem_line_idx = 5 if len(lines) > 6 else 4
        count_line_idx = elem_line_idx + 1
        
        elements = lines[elem_line_idx].split()
        atom_counts = list(map(int, lines[count_line_idx].split()))
        
        # Compute atomic-number ratios (fractions, 0-1), keeping only preset elements
        total_atoms = sum(atom_counts)
        for elem in required_elements:
            if elem in elements:
                idx = elements.index(elem)
                element_ratio[f'x({elem})'] = atom_counts[idx] / total_atoms  # use fraction instead
            else:
                element_ratio[f'x({elem})'] = 0.0
        
        return element_ratio
    except Exception as e:
        print(f"Failed to parse POSCAR {poscar_path}: {e}")
        # Return default values to avoid interrupting the program
        return {f'x({elem})': 0.0 for elem in required_elements}

def extract_physical_values(filename, phase):
    """
    Extract physical-property indices from a POSCAR filename and map them to actual values; also return the second number (element-combination identifier)
    Filename format: POSCAR_30199_53.02_35.44 ->
        second number: 30199 (element-combination identifier)
        bulk modulus index: 53.02, lattice thermal conductivity index: 35.44
    """
    # Regex matching the numeric parts: group 1 = second number, group 2 = bulk modulus index, group 3 = lattice thermal conductivity index
    pattern = r'POSCAR_(\d+)_([\d\.]+)_([\d\.]+)'
    match = re.search(pattern, filename)
    if not match:
        return 0, 0.0, 0.0  # (element-combination identifier, bulk modulus value, lattice thermal conductivity value)
    
    # Extract the second number (element-combination identifier)
    elem_group_id = int(match.group(1))
    bm_idx = float(match.group(2))
    ltc_idx = float(match.group(3))
    
    # Select the mapping space according to the phase
    bm_subspace = "bm_quinary_fcc" if phase == "fcc" else "bm_quinary_bcc"
    ltc_subspace = "ltc_quinary_fcc" if phase == "fcc" else "ltc_quinary_bcc"
    
    # Map to actual values
    bm_value = value_mapping(bm_idx, bm_subspace)
    ltc_value = value_mapping(ltc_idx, ltc_subspace)
    
    return elem_group_id, bm_value, ltc_value

def load_samples_from_poscar(phase):
    """Load all POSCAR data of the given phase and parse them into a sample list (deduplicated by element-combination ID)"""
    # Define all paths
    paths = {
        "fcc": {
            "normal": ["./data/quinary_fcc/"],  # TODO: set your local data path
            "key": "./data/fcc/"  # TODO: set your local data path
        },
        "bcc": {
            "normal": [
                "./data/quinary_bcc/",  # TODO: set your local data path
                "./data/quinary_bcc_2/"  # TODO: set your local data path
            ],
            "key": "./data/bcc/"  # TODO: set your local data path
        }
    }
    
    samples = []
    key_samples = []
    sample_counter = 1
    # Track element-combination IDs already seen (deduplication)
    seen_elem_group_ids = set()

    # Load normal samples (non-key)
    for dir_path in paths[phase]["normal"]:
        if not os.path.exists(dir_path):
            print(f"Path does not exist, skipping: {dir_path}")
            continue
        
        for filename in os.listdir(dir_path):
            if filename.startswith("POSCAR"):
                poscar_path = os.path.join(dir_path, filename)
                # Extract element-combination ID and physical properties
                elem_group_id, bm_val, ltc_val = extract_physical_values(filename, phase)
                
                # Deduplication logic: keep only the first occurrence of each element-combination ID
                if elem_group_id in seen_elem_group_ids:
                    print(f"Skipping duplicate element combination {elem_group_id}: {filename}")
                    continue
                seen_elem_group_ids.add(elem_group_id)
                
                # Parse element ratios of the corresponding phase
                elem_ratio = parse_poscar(poscar_path, phase)
                
                sample = {
                    'Sample': f'S{sample_counter}',
                    'ElemGroupID': elem_group_id,  # keep the element-combination ID for easy cross-checking
                    **elem_ratio,
                    'BM': bm_val,  # bulk modulus
                    'LTC': ltc_val  # lattice thermal conductivity
                }
                samples.append(sample)
                sample_counter += 1

    # Load key samples (Key): key samples take priority; if a normal sample already has the same ID, replace it
    key_dir = paths[phase]["key"]
    if os.path.exists(key_dir):
        key_counter = 1
        for filename in os.listdir(key_dir):
            if filename.startswith("POSCAR"):
                poscar_path = os.path.join(key_dir, filename)
                # Extract element-combination ID and physical properties
                elem_group_id, bm_val, ltc_val = extract_physical_values(filename, phase)
                
                # If the Key sample's ID already exists, first remove the corresponding entry from the normal samples
                if elem_group_id in seen_elem_group_ids:
                    print(f"Key sample replacing normal sample {elem_group_id}: {filename}")
                    # Filter out the entry with the same ID from the normal samples
                    samples = [s for s in samples if s['ElemGroupID'] != elem_group_id]
                
                seen_elem_group_ids.add(elem_group_id)
                # Parse element ratios of the corresponding phase
                elem_ratio = parse_poscar(poscar_path, phase)
                
                key_sample = {
                    'Sample': f'Key{key_counter}',
                    'ElemGroupID': elem_group_id,  # keep the element-combination ID for easy cross-checking
                    **elem_ratio,
                    'BM': bm_val,
                    'LTC': ltc_val
                }
                key_samples.append(key_sample)
                key_counter += 1

    # Merge all samples
    samples.extend(key_samples)
    print(f"Loaded {len(samples)} samples after deduplication (normal: {len(samples)-len(key_samples)}, Key: {len(key_samples)})")
    return samples

# ---------------------- Main program execution ----------------------
if __name__ == "__main__":
    # Select the phase to plot (fcc / bcc)
    #phase = "fcc"
    phase = "bcc"

    # Load sample data (deduplicated)
    samples = load_samples_from_poscar(phase)
    if not samples:
        print("No sample data loaded!")
    else:
        df = pd.DataFrame(samples)
        # Define plot columns (element ratios of the phase + two physical properties)
        element_columns = [f'x({elem})' for elem in PHASE_ELEMENTS[phase]]
        columns = element_columns + ['BM', 'LTC']
        n_cols = len(columns)
        
        # Define the Y-axis range for each column
        col_ranges = []
        for col in columns:
            if col.startswith('x('):  # element-ratio column
                col_ranges.append(Y_AXIS_RANGES["element_ratio"])
            elif col == 'BM':       # bulk-modulus column
                col_ranges.append(Y_AXIS_RANGES["bm"])
            elif col == 'LTC':       # lattice-thermal-conductivity column
                if phase == 'bcc':
                    col_ranges.append(Y_AXIS_RANGES["ltc_bcc"])
                else:
                    col_ranges.append(Y_AXIS_RANGES["ltc_fcc"])

        # ---------------------- Plot configuration ----------------------
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 18,
            'axes.unicode_minus': False
        })
        fig, ax = plt.subplots(figsize=(14, 7))  # enlarge the canvas to fit many elements

        # Normalize data to the 0-1 range for plotting (unified visualization across different Y-axis ranges)
        def normalize_value(val, val_range):
            """Normalize a value to the 0-1 range"""
            min_val, max_val = val_range
            return (val - min_val) / (max_val - min_val) if max_val > min_val else 0

        # Store normalized values of normal samples for drawing boxes
        normal_vals_list = []
        # 1. Plot normal samples (faint red)
        normal_mask = ~df['Sample'].str.startswith('Key')
        for idx, row in df[normal_mask].iterrows():
            norm_values = []
            for val, rng in zip(row[columns].values, col_ranges):
                norm_values.append(normalize_value(val, rng))
            normal_vals_list.append(norm_values)  # save normalized values
            ax.plot(range(n_cols), norm_values, color='#FF0000', alpha=0.2, linewidth=0.8)

        # 2. Plot key samples (highlighted red)
        key_mask = df['Sample'].str.startswith('Key')
        key_vals_list = []
        for idx, row in df[key_mask].iterrows():
            norm_values = []
            for val, rng in zip(row[columns].values, col_ranges):
                norm_values.append(normalize_value(val, rng))
            key_vals_list.append(norm_values)
            ax.plot(range(n_cols), norm_values, color='#FF0000', alpha=1.0, linewidth=2.5)

        # ---------------------- Style adjustments ----------------------
        # Hide the frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Set the overall Y-axis range (0-1 after normalization)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_visible(False)

        # Draw the vertical axis, custom tick marks and tick labels for each variable
        for i in range(n_cols):
            col_name = columns[i]
            val_range = col_ranges[i]
            min_val, max_val = val_range
            # Draw the vertical axis line
            ax.axvline(x=i, color='black', linewidth=0.8)
            
            # Generate ticks for this column (5 equally spaced ticks)
            ticks = np.linspace(min_val, max_val, 5)
            norm_ticks = [normalize_value(t, val_range) for t in ticks]
            
            # Draw tick marks and tick labels
            for tick, norm_tick in zip(ticks, norm_ticks):
                # Short tick mark (horizontal)
                ax.plot([i - 0.1, i], [norm_tick, norm_tick], color='black', linewidth=0.8)
                # Tick label (1 decimal place)
                ax.text(i - 0.22, norm_tick, f'{tick:.1f}', fontsize=18, ha='right', va='center')

        # X-axis label settings
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(columns, fontsize=18, rotation=45, ha='right')  # smaller font + rotation to avoid overlap
        ax.tick_params(axis='x', length=0)  # hide X-axis tick marks
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Draw light-gray boxes for non-key samples (ignoring zero values)
        if normal_vals_list:
            for i in range(n_cols):
                # Keep only non-zero values
                normal_vals_on_axis = [vals[i] for vals in normal_vals_list if vals[i] != 0]
                
                if len(normal_vals_on_axis) == 0:
                    continue  # this column is all zeros; do not draw a box
                    
                min_val = min(normal_vals_on_axis) - 0.05
                max_val = max(normal_vals_on_axis) + 0.05
                # Light-gray box (alpha=0.3 adds transparency to avoid occlusion, linewidth=1.5 line width)
                rect = plt.Rectangle((i-0.2, min_val), 0.4, max_val-min_val, 
                                    linewidth=1.5, edgecolor='#D3D3D3', facecolor='none', alpha=0.8)
                ax.add_patch(rect)

        # Blue boxes marking key systems (ignoring zero values)
        if key_vals_list:
            for i in range(n_cols):
                # Keep only non-zero values
                key_vals_on_axis = [vals[i] for vals in key_vals_list if vals[i] != 0]
                
                if len(key_vals_on_axis) == 0:
                    continue  # this column is all zeros; do not draw a box
                    
                min_val = min(key_vals_on_axis) - 0.05
                max_val = max(key_vals_on_axis) + 0.05
                rect = plt.Rectangle((i-0.2, min_val), 0.4, max_val-min_val, 
                                    linewidth=2, edgecolor='blue', facecolor='none')
                ax.add_patch(rect)

        plt.tight_layout()
        plt.show()
