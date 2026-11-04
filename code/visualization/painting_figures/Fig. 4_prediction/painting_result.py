import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
#plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16

property_name = "shear_modulus"

result_data = "./data/Fig. 4_prediction/"  # TODO: set your local data path

if property_name == "bulk_modulus":
    x_observed = torch.load(result_data +  property_name + "/" + "x_observed-0-0.879.pt")
    y_predicted = torch.load(result_data +  property_name + "/" + "y_predicted-0-0.879.pt")
elif property_name == "shear_modulus":
    x_observed_raw = torch.load(result_data +  property_name + "/" +  "x_observed-0-0.532.pt")
    y_predicted_raw = torch.load(result_data  + property_name + "/" +  "y_predicted-0-0.532.pt")
    x_observed = []
    y_predicted = []
    for i in range(len(x_observed_raw)):
        if abs(x_observed_raw[i]) > 1e-5:
            x_observed.append(x_observed_raw[i])
            y_predicted.append(y_predicted_raw[i])
elif property_name == "lattice_thermal_conduct":
    x_observed = torch.load(result_data +  property_name + "/" +  "x_observed-0-0.871.pt")
    y_predicted = torch.load(result_data  + property_name + "/" +  "y_predicted-0-0.871.pt")
    
data = pd.DataFrame({'x_observed':x_observed, 'y_predicted': y_predicted})
r_square = r2_score(x_observed,y_predicted)
# Set up the canvas
plt.figure(figsize=(6, 6),dpi = 600)

# Plot the scatter of observed vs predicted values
if property_name == "bulk_modulus":
    sns.scatterplot(data = data, x = "x_observed",y="y_predicted", color='#3E608D', label = "Validation set R² : {:.4f}".format(r_square))
elif property_name == "shear_modulus":
    sns.scatterplot(data = data, x = "x_observed",y="y_predicted", color='#7B8C9D', label = "Validation set R² : {:.4f}".format(r_square))
elif property_name == "lattice_thermal_conduct":
    sns.scatterplot(data = data, x = "x_observed",y="y_predicted", color='#CB9475', label = "Validation set R² : {:.4f}".format(r_square))
dot = [round(x * 0.05, 2) for x in range(int(max(x_observed) / 0.05) + 1)]
plt.plot(dot, dot, color='#8D2F25', linestyle='--')  # perfect prediction line
plt.ylim(0,max(x_observed))
plt.xlim(0,max(x_observed))

# Add labels
if property_name == "bulk_modulus":
    plt.title('Bulk Modulus in MP')
    #plt.xlabel('Observed Values / log(GPa)')
    #plt.ylabel('Predicted Values / log(GPa)')
    plt.xlabel('Observed Values [log(GPa)]')
    plt.ylabel('Predicted Values [log(GPa)]')
elif property_name == "shear_modulus":
    plt.title('Shear Modulus in MP')
    #plt.xlabel('Observed Values / log(GPa)')
    #plt.ylabel('Predicted Values / log(GPa)')
    plt.xlabel('Observed Values [log(GPa)]')
    plt.ylabel('Predicted Values [log(GPa)]')
elif property_name == "lattice_thermal_conduct":
    plt.title('Lattice Thermal Conductivity in AFLOW')
    #plt.xlabel('Observed Values / log(W·m⁻¹·K⁻¹)')
    #plt.ylabel('Predicted Values / log(W·m⁻¹·K⁻¹)')
    #plt.xlabel(r'Observed Values / log($\mathrm{W \cdot m^{-1} \cdot K^{-1}}$)')
    #plt.ylabel(r'Predicted Values / log($\mathrm{W \cdot m^{-1} \cdot K^{-1}}$)')
    plt.xlabel(r'Observed Values [log($\mathrm{W \cdot m^{-1} \cdot K^{-1}}$)]')
    plt.ylabel(r'Predicted Values [log($\mathrm{W \cdot m^{-1} \cdot K^{-1}}$)]')
"""
if property_name == "bulk_modulus":
    plt.savefig("./bulk_modulus_in_MP.png",dpi =600)
elif property_name == "shear_modulus":
    plt.savefig("./shear_modulus_in_MP.png",dpi =600)
elif property_name == "lattice_thermal_conduct":
    plt.savefig("./lattice_thermal_conduct_in_AFLOW.png",dpi =600)
"""
plt.legend()
#plt.grid()

# Display the plot
plt.show()