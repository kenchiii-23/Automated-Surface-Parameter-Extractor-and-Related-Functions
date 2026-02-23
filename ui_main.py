from customtkinter import *
from tkinter import *
from tkinter.filedialog import askopenfilename
import pandas as pd
import os
import main as mn
import rasterio
from rasterio.plot import show
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==================== Variables ==================== #
cbDim = 18
cbBorder = 2

# ==================== Functions ==================== #
def rasterViewing(rasterPath, cityList):
    # Get Province Name
    provinceName=cityList[0]
    cityName=cityList[1]
    
    # Set Popup Title
    rasterView = CTk()
    rasterView.title(f"{provinceName}, {cityName} Raster View")

    # Add underscore for valid path naming
    provinceName = provinceName.replace(' ', '_')
    cityName=cityName.replace(' ', '_')

    data_path = rasterPath + fr"\OutputRaster\{provinceName}\{cityName}"
    print(data_path)
    data_list = []
    values = ["Aspect", "Distance to River", 
            "Elevation", "Flow Accumulation", 
            "Flow Direction", "Planform Curvature", 
            "Profile Curvature", "Relative Relief", 
            "Slope", "TWI"]

    cmap_list = ['twilight', 'Blues_r', 
                'terrain', 'inferno', 
                'tab10', 'coolwarm', 
                'RdBu_r', 'cividis', 
                'magma', 'viridis']

    bar_name = ['Degrees (°)', 'meters (m)', 
                'meters (m)', 'square meters (m²)', 
                ' ', '1/m (per meter)', 
                '1/m (per meter)', 'meters (m)', 
                'Degrees (°)', ' ']

    norm_list = [cl.Normalize(vmin=None, vmax=None),cl.Normalize(vmin=None, vmax=None),
                cl.Normalize(vmin=None, vmax=None),cl.LogNorm(),
                cl.LogNorm(),cl.TwoSlopeNorm(vcenter=0),
                cl.TwoSlopeNorm(vcenter=0),cl.Normalize(vmin=None, vmax=None),
                cl.Normalize(vmin=None, vmax=None),cl.Normalize(vmin=None, vmax=None)]

    # Adds each .tif file in the directory
    for file_path in Path(data_path).glob('*.tif'):
        if file_path.is_file():
            data_list.append(file_path)

    rasterSelection = StringVar(value="Aspect")
    def rasterSelected(choice):

        selection_index = values.index(rasterSelection.get())
        data = data_list[selection_index]

        tiff = rasterio.open(data)

        fig, ax = plt.subplots(figsize=(8,6), dpi=120)


        ax.set_title(values[selection_index],
                fontsize=14,
                fontweight='bold',
                pad=12)
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.tick_params(labelsize=9)

        show(tiff, ax=ax, cmap=cmap_list[selection_index], norm=norm_list[selection_index])

        cbar = fig.colorbar(ax.images[0],
                        ax=ax,
                        shrink=0.85,
                        pad=0.02)

        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(bar_name[selection_index], fontsize=11)
        #ax.set_axis_off()

        canvas = FigureCanvasTkAgg(fig, master=rasterView)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=2, column=0, sticky="nsew")  

        
    label = CTkLabel(rasterView, text=f"{cityList[1]}, {cityList[0]} Rasters", font=CTkFont(size=16, weight="bold"))
    optionMenu_s1_1 = CTkOptionMenu(rasterView, values=values, variable=rasterSelection, command=rasterSelected)

    label.grid(row=0, column=0, padx=10, pady=10)
    optionMenu_s1_1.grid(row=1, column=0, padx=10, pady=10)

    rasterView.mainloop()

def popupDialog(message="Default text"):
    popup = CTkToplevel(root)
    popup.title("Notice")
    popup.geometry("300x150")
    popup.resizable(False, False)

    popup.attributes("-topmost", True)
    popup.grab_set()  # Make modal

    frame = CTkFrame(popup, corner_radius=10)
    frame.pack(expand=True, fill="both", padx=10, pady=10)

    label = CTkLabel(
        frame,
        text= message,
        wraplength=300,
        justify="center",
        font=CTkFont(size=14)
    )
    label.pack(pady=(25, 15))

    ok_button = CTkButton(
        frame,
        text="OK",
        width=100,
        command=popup.destroy
    )
    ok_button.pack(pady=(0, 20))


# ==================== Main View ==================== #

# *** Widget Name Syntaxis: widgetType_s[sectionNumber]_*optional-b[batchNumber]_[widgetNumber]

root = CTk()
root.title("Surface Parameter Extractor")
set_default_color_theme("core/theme.json")
tabview = CTkTabview(master=root, width=450, height=300)
tabview.grid(row=0, column=0, padx=20, pady=20, ipadx=10, ipady=10)

section1 = tabview.add("1")  
section2 = tabview.add("2")  
section3 = tabview.add("3")  
section4 = tabview.add("4")
tabview.set("1")  # set currently visible tab


############### Section 1: Input DEM ###############

#section1 = CTkFrame(root, width=350, height=250)
label_s1_1 = CTkLabel(section1, text="1. Input DEM", font=CTkFont(size=20, weight="bold"))

## Creating Radio Buttons
def radio_s1_b1_event():
    rad = radio_s1_b1_var.get()
    if rad == 1:
        button_s1_1.configure(state=DISABLED)
        entry_s1_1.configure(state=DISABLED)
        optionmenu_s1_1.configure(state=NORMAL)

    elif rad ==2:
        entry_s1_1.configure(state=NORMAL)
        button_s1_1.configure(state=NORMAL)
        optionmenu_s1_1.configure(state=DISABLED)

radio_s1_b1_var = IntVar(value=1)
radio_s1_b1_1 = CTkRadioButton(section1, text="Choose from Built-in DEM", command=radio_s1_b1_event, 
                               variable= radio_s1_b1_var, value=1,
                               radiobutton_height=cbDim, radiobutton_width=cbDim,
                               border_width_checked=cbBorder+2, border_width_unchecked=cbBorder)

radio_s1_b1_2 = CTkRadioButton(section1, text="Import Custom DEM", command=radio_s1_b1_event, 
                               variable= radio_s1_b1_var, value=2,
                               radiobutton_height=cbDim, radiobutton_width=cbDim,
                               border_width_checked=cbBorder+2, border_width_unchecked=cbBorder)

## Creating Option Menu and Button
def optionmenu_s1_1_callback(choice):
    print("optionmenu dropdown clicked:", optionmenu_s1_1_var.get())

optionmenu_s1_1_var = StringVar(value="NASA SRTM 90m (Default)")
optionmenu_s1_1 = CTkOptionMenu(section1, values=["NASA SRTM 90m (Default)"], variable=optionmenu_s1_1_var, command=optionmenu_s1_1_callback)


def button_s1_1_event():
        global DEM_filepath, DEM_filename
        Tk().withdraw()
        dir = askopenfilename()
        if not dir.__contains__(".asc"):
            popupDialog(f"File is in incorrect file format")
            radio_s1_b1_var.set(1)
            radio_s1_b1_event()

        else:
            DEM_filename = os.path.basename(dir) # Filename with extension
            DEM_filepath.set(dir) 


DEM_filepath = StringVar()
entry_s1_1 = CTkEntry(section1, placeholder_text="Enter DEM Path", width=300, textvariable=DEM_filepath)
button_s1_1 = CTkButton(section1, text="Choose", command=button_s1_1_event, width=50)


# Disable option menu and button at the start
button_s1_1.configure(state=DISABLED)
entry_s1_1.configure(state=DISABLED)

## Placing widgets in the frame, section1
label_s1_1.grid(row=0, column=0, columnspan=2, pady=10, padx=10)
radio_s1_b1_1.grid(row=1, column=0, columnspan=2,pady=10, padx=10, sticky="w")
optionmenu_s1_1.grid(row=2, column=0, columnspan=2,pady=(0,10), padx=10)
radio_s1_b1_2.grid(row=3, column=0, columnspan=2,pady=(0,10), padx=10, sticky="w")
entry_s1_1.grid(row=4, column=0, pady=(0,10), padx=(10,2))
button_s1_1.grid(row=4, column=1, pady=(0,10), padx=(2,10))
#section1.grid(row=0, column=0, pady=10, padx=10)
section1.grid_propagate(False)
section1.grid_rowconfigure(5, weight=1)
section1.grid_columnconfigure(0, weight=1)



############### Section 2: Seclect City ###############
#section2 = CTkFrame(root, width=350, height=250)
label_s2_1 = CTkLabel(section2, text="2. Select City", font=CTkFont(size=20, weight="bold"))

## Creating Labels
label_s2_2 = CTkLabel(section2, text="Province")
label_s2_3 = CTkLabel(section2, text="Municipality")

## Creating Option Menu
base_dir = os.getcwd()
citiesDB = pd.read_csv(f"{base_dir}/data/raw/dependencies/cities/cities.csv")
provinces_list = citiesDB['NAME_1'].drop_duplicates().tolist()
municipality_list = citiesDB[citiesDB['NAME_1'] == provinces_list[0]]['NAME_2'].tolist()

selectedProvince = StringVar(value=provinces_list[0])
selectedMunicipality = StringVar(value=municipality_list[0])

def optionmenu_s2_1_event(choice):
    global selectedProvince, municipality_list, optionmenu_s2_2, selectedMunicipality
    municipality_list = citiesDB[citiesDB['NAME_1'] == selectedProvince.get()]['NAME_2'].tolist()

    optionmenu_s2_2.configure(values=municipality_list)
    selectedMunicipality.set(municipality_list[0])


optionmenu_s2_1 = CTkOptionMenu(section2, values=provinces_list, command=optionmenu_s2_1_event, variable=selectedProvince)
optionmenu_s2_2 = CTkOptionMenu(section2, values=municipality_list, variable=selectedMunicipality)

## PreSelect CDO
selectedProvince.set(provinces_list[48])
optionmenu_s2_1_event(provinces_list[48])
selectedMunicipality.set(municipality_list[4])

## Placing widgets in the frame, section2
label_s2_1.grid(row=0, column=0, pady=10, padx=10)
label_s2_2.grid(row=1, column=0, padx=10, sticky="w")
optionmenu_s2_1.grid(row=2, column=0, padx=10)
label_s2_3.grid(row=3, column=0, padx=10, pady=(10,0), sticky="w")
optionmenu_s2_2.grid(row=4, column=0, padx=10, pady=(0,10))
#section2.grid(row=0, column=1, pady=10, padx=10)
section2.grid_propagate(False)
section2.grid_rowconfigure(5, weight=1)
section2.grid_columnconfigure(0, weight=1)



############### Section 3: Point Extraction Setting ###############
#section4 = CTkFrame(root, width=350, height=250)
label_s3_1 = CTkLabel(section3, text="3. Point Extraction Setting", font=CTkFont(size=20, weight="bold"))


## Create Checkbox
def checkbox_s3_1_event():
    checked = checkbox_s3_1_var.get()
    if checked == "on":
        for widget in [entry_s3_1, button_s3_1, entry_s3_2, button_s3_2]:
            widget.configure(state=NORMAL)
            root.update_idletasks()
    else:
        for widget in [entry_s3_1, button_s3_1, entry_s3_2, button_s3_2]:
            widget.configure(state=DISABLED)
            root.update_idletasks()

checkbox_s3_1_var = StringVar(value="off")
checkbox_s3_1 = CTkCheckBox(section3, 
                            text="Extract parameters at specified points", 
                            height=10, 
                            variable=checkbox_s3_1_var,
                            checkbox_height=cbDim, checkbox_width=cbDim, 
                            border_width=cbBorder, corner_radius=cbBorder,
                            onvalue="on", offvalue="off", command=checkbox_s3_1_event)


## Create Entry and Buttons
inputPointCSV_filepath=StringVar()
entry_s3_1 = CTkEntry(section3, placeholder_text="Input points .csv path", width=300, textvariable=inputPointCSV_filepath)
entry_s3_1.configure(state=DISABLED)


def button_s3_1_event():
        global inputPointCSV_filepath
        dir = askopenfilename()
        if not dir.__contains__(".csv"): 
            popupDialog(f"File selected is not CSV")
            inputPointCSV_filepath.set("")
            checkbox_s3_1_var.set("off")
            checkbox_s3_1_event()
        else:
             inputPointCSV_filepath.set(dir)

button_s3_1 = CTkButton(section3, text="Choose", command=button_s3_1_event, width=50)
button_s3_1.configure(state=DISABLED)

outputPointsParameter_filepath=StringVar()
entry_s3_2 = CTkEntry(section3, placeholder_text="Output point parameter path", width=300, textvariable=outputPointsParameter_filepath)
entry_s3_2.configure(state=DISABLED)


def button_s3_2_event():
        global outputPointsParameter_filepath
        dir = filedialog.askdirectory()
        outputPointsParameter_filepath.set(dir)

button_s3_2 = CTkButton(section3, text="Choose", command=button_s3_2_event, width=50)
button_s3_2.configure(state=DISABLED)


## Placing elements in section 3
label_s3_1.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
checkbox_s3_1.grid(row=1, column=0,columnspan=2, padx=10, pady=10, sticky="w")
entry_s3_1.grid(row=2, column=0, padx=(10,2), pady=(0,10))
button_s3_1.grid(row=2, column=1, padx=(2,10), pady=(0,10))
entry_s3_2.grid(row=3, column=0, padx=(10,2), pady=(0,10))
button_s3_2.grid(row=3, column=1, padx=(2,10), pady=(0,10))

section3.grid_propagate(False)
section3.grid_rowconfigure(5, weight=1)
section3.grid_columnconfigure(0, weight=1)






############### Section 4: Configuration ###############
#section3 = CTkFrame(root, width=350, height=250)
label_s4_1 = CTkLabel(section4, text="4. Configuration", font=CTkFont(size=20, weight="bold"))

## Creating Checkboxes
checkbox_s4_1_var = StringVar(value="off")
checkbox_s4_1 = CTkCheckBox(section4, 
                            text="Load Previous DEM Configuration if present", 
                            height=10, 
                            variable=checkbox_s4_1_var,
                            checkbox_height=cbDim, checkbox_width=cbDim, 
                            border_width=cbBorder, corner_radius=cbBorder,
                            onvalue="on", offvalue="off")

checkbox_s4_2_var = StringVar(value="off")
checkbox_s4_2 = CTkCheckBox(section4, 
                            text="Load Previous Parameters if present", 
                            height=10, 
                            variable=checkbox_s4_2_var,
                            checkbox_height=cbDim, checkbox_width=cbDim, 
                            border_width=cbBorder, corner_radius=cbBorder,
                            onvalue="on", offvalue="off")
def checkbox_s4_3_event():
    checked = checkbox_s4_3_var.get()
    if checked == "on":
        entry_s4_1.configure(state=NORMAL)
        button_s4_1.configure(state=NORMAL)
    else:
        entry_s4_1.configure(state=DISABLED)
        button_s4_1.configure(state=DISABLED)

checkbox_s4_3_var = StringVar(value="off")
checkbox_s4_3 = CTkCheckBox(section4, 
                            text="Save output rasters", 
                            height=10, 
                            variable=checkbox_s4_3_var,
                            checkbox_height=cbDim, checkbox_width=cbDim, 
                            border_width=cbBorder, corner_radius=cbBorder,
                            onvalue="on", offvalue="off", command=checkbox_s4_3_event)

saveRasterPath = StringVar()
entry_s4_1 = CTkEntry(section4, placeholder_text="Ouput raster path", width=300, textvariable=saveRasterPath)
entry_s4_1.configure(state=DISABLED)

def button_s4_1_event():
        global saveRasterPath
        dir = filedialog.askdirectory()
        saveRasterPath.set(dir)

button_s4_1 = CTkButton(section4, text="Choose", command=button_s4_1_event, width=50)
button_s4_1.configure(state=DISABLED)

## Placing elements in frame section 3
label_s4_1.grid(row=0, column=0, columnspan=2, pady=10, padx=10)
checkbox_s4_1.grid(row=1, column=0,columnspan=2, padx=10, pady=(0,10), sticky="w")
checkbox_s4_2.grid(row=2, column=0,columnspan=2, padx=10, pady=(0,10), sticky="w")
checkbox_s4_3.grid(row=3, column=0, columnspan=2,padx=10, pady=(0,10), sticky="w")
entry_s4_1.grid(row=4, column=0, padx=(10,2), pady=0)
button_s4_1.grid(row=4, column=1, padx=(2,10), pady=0)
#section3.grid(row=1, column=0, padx=10, pady=10)
section4.grid_propagate(False)
section4.grid_rowconfigure(5, weight=1)
section4.grid_columnconfigure(0, weight=1)

demPath=""
inputCSVPath=""
outputCSVPath=""
loadConfig=False
loadParams=False
saveRaster=False
extractPoints=False
loadFromFolder=False 
demFolder = ""
city = []

def button():
        
        try: 
            assignVariables()
        except:
            popupDialog("Error assigning initial variables")

        print(f"DEM Path: {demPath}")
        print(f"Load Config: {loadConfig}")
        print(f"Load Parameters: {loadParams}")
        print(f"Save Raster: {saveRaster}")
        print(f"Output Raster Path: {saveRasterPath.get()}")
        print(f"Extract Points: {extractPoints}")
        print(f"Input CSV File: {inputCSVPath}")
        print(f"Output CSV Path: {outputCSVPath}")
        print(f"City: {city}")
        print(f"Load from Folder: {loadFromFolder}")
        print(f"DEM Folder: {demFolder}")

        try:
            mn.call(demPath, loadConfig, loadParams, saveRaster, saveRasterPath.get(), extractPoints, 
                    inputCSVPath, outputCSVPath, city,
                    loadFromFolder, demFolder)
            
            popupDialog(f"Successfully calculated parameters for {city[1]}")
            if saveRaster:
                 rasterViewing(saveRasterPath.get(), city)

        except FileNotFoundError:
             print("File not Found")

        except ValueError:
             popupDialog("Chosen city is out of bounds to the selected DEM")

def assignVariables():
    ## 1.) Input DEM
    global demPath, DEM_filepath, demFolder, loadFromFolder
    if radio_s1_b1_var.get() == 1:
        if optionmenu_s1_1_var.get() == "NASA SRTM 90m (Default)":
            demPath = f"{base_dir}/data/raw/dependencies/NASA SRTM"
            demFolder = demPath
            loadFromFolder = True            
    else:
        loadFromFolder = False
        if not os.path.isfile(DEM_filepath.get()):
            print(f"{DEM_filepath.get()}.....error")
            popupDialog("DEM Path is invalid")
            return False
        else:
            print(f"DEM: {DEM_filepath.get()}")
            demPath = DEM_filepath.get()
            print(f"DEM ASS: {demPath}")

    ## 2.) City Selection
    global city
    city = [selectedProvince.get(), selectedMunicipality.get()]

    ## 3.) Point Extraction Setting
    global inputCSVPath, outputCSVPath, extractPoints
    extractPoints = False
    
    if checkbox_s3_1_var.get() == "on":
        global outputCSVPath, inputCSVPath
        extractPoints = True
        # Check if input path exists
        if not os.path.isfile(inputPointCSV_filepath.get()):
            print(f"Input is not file")
            popupDialog("Points CSV file is invalid")
            return
        # Check if input file is CSV
        elif not inputPointCSV_filepath.get().__contains__(".csv"):
            print(f"Input is not csv")
            popupDialog("Points CSV file is invalid")
            return
        else:
            inputCSVPath = inputPointCSV_filepath.get()
        
        #Check if output path exist
        if not os.path.isdir(outputPointsParameter_filepath.get()):
            popupDialog("Output path for points parameter is invalid")
        else:
            outputCSVPath = outputPointsParameter_filepath.get()

    ## 4.) Configuration
    global loadConfig, loadParams, saveRaster, saveRasterPath
    loadConfig=True if checkbox_s4_1_var.get()== "on" else False
    loadParams=True if checkbox_s4_2_var.get()== "on" else False
    saveRaster=True if checkbox_s4_3_var.get()== "on" else False
    if saveRaster:
        if not os.path.isdir(saveRasterPath.get()):
            popupDialog("Output path for rasters is invalid")
            return False


button = CTkButton(root, text="Run Main Program", command=button)
button.grid(row=2, column=0, pady=10)


root.mainloop()



