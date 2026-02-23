import geopandas as gpd
import numpy as np
from shapely.geometry import box
import rasterio
from rasterio.mask import mask
from rasterio import features
from rasterio.merge import merge
from scipy.ndimage import distance_transform_edt
import os


def  main():
    # --- 1. Load shapefile and define raster bounds ---
    base = r"C:\Users\KennethBaluyos (EPA)\Documents\04 - Manifesting RSTF\Projects\Project Reworked"
    shpPath = rf"{base}\data\raw\dependencies\NAMRIA_RiverNetwork\phl_rivl_250k_NAMRIA.shp"

    cutShpFromCity(shpPath, "Iligan City", "Rivers")


def cutShpFromCity(shpPath, cityName, shpName):
    # Load both shapefiles
    base = os.getcwd()
    city_path = rf"{base}\data\raw\dependencies\cities\cities.shp"


    # Load shapefiles
    shp = gpd.read_file(shpPath)
    city_boundary = gpd.read_file(city_path)

    # (Optional) If your city shapefile has many cities, pick one by name
    city = city_boundary[city_boundary["NAME_2"] == cityName]

    # Ensure CRS match
    shp = shp.to_crs(city.crs)

    # Clip rivers to city boundary
    shp_clip = gpd.overlay(shp, city, how='intersection')

    # Save result
    shp_clip.to_file(rf"{base}\data\raw\{cityName}_{shpName}.shp")


def cutAsciFromCity(AsciiPath, cityList, openFromFolder=False, AsciiFolder=None):
    provinceName=cityList[0]
    provinceName = provinceName.replace(' ', '_')
    cityName=cityList[1]
    cityName =  cityName.replace(' ', '_')

    
    
    print(f"Clipping ASCII DEM for {cityName}")
    base = os.getcwd()
    shp_path = rf"{base}\data\raw\dependencies\cities\cities.shp"


    # --- 2. Load shapefile ---
    shp = gpd.read_file(shp_path)

    # --- 6. Select target city ---
    city = shp[(shp["NAME_1"] == cityList[0]) & (shp["NAME_2"] == cityList[1])]
    if city.empty:
        print("City not found!")
        return
    
    
        
    if openFromFolder:
        indexPath = AsciiFolder + r"\index.geojson"   

        dem_index = gpd.read_file(indexPath)
        city = city.to_crs(dem_index.crs)
        intersecting = dem_index[dem_index.intersects(city.geometry.iloc[0])]
        intersecting_dems = intersecting["filename"].tolist()

        """if file.endswith(".asc"):
            path = os.path.join(AsciiFolder, file)

            with rasterio.open(path) as src:
                dem_bbox = box(*src.bounds)

                # Check intersection
                if city_bbox.intersects(dem_bbox):
                    intersecting_dems.append(path) """

        
        if not intersecting_dems:
            print("No DEM tiles intersect this city.")
            return
        
        elif len(intersecting_dems) == 1: 
            with rasterio.open(intersecting_dems[0]) as src:
                out_image, out_transform = mask(src, city.geometry, crop=True)
                out_meta = src.meta.copy()

            # --- 8. Update metadata for ASCII output ---
            out_meta.update({
                "driver": "AAIGrid",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # --- 9. Save as new ASCII ---
            ascii_out = rf"{base}\data\temp\{provinceName}\{cityName}\asc\{cityName}.asc"
            print(f"Saving Clipped DEM as: {ascii_out}")
            os.makedirs(os.path.dirname(ascii_out), exist_ok=True)

            with rasterio.open(ascii_out, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f"Clipped ASCII DEM for {cityName}, saved as: {ascii_out}")
            return ascii_out
        


        print("Intersecting DEMs:", intersecting_dems)

        # Open only intersecting DEMs
        src_files = [rasterio.open(p) for p in intersecting_dems]

        # Ensure CRS matches
        city = city.to_crs(src_files[0].crs)

        
        # Merge only intersecting DEMs
        print("Merging DEMs")
        mosaic, out_transform = merge(src_files)

        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "count": 1
        })

        print("Writing to Memory and Clipping")
        # Write to memory and clip
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**out_meta) as dataset:
                dataset.write(mosaic)

                clipped, clipped_transform = mask(
                    dataset,
                    city.geometry,
                    crop=True
                )

        # Update metadata after clipping
        out_meta.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": clipped_transform
        })

        # Save output
        ascii_out = rf"{base}\data\temp\{provinceName}\{cityName}\asc\{cityName}.asc"
        print(f"Saving Clipped DEM as: {ascii_out}")
        os.makedirs(os.path.dirname(ascii_out), exist_ok=True)

        with rasterio.open(ascii_out, "w", **out_meta) as dest:
            dest.write(clipped)

        print("City DEM extracted successfully!")

        for src in src_files:
            src.close()

        return ascii_out


    else:
        # --- 3. Load DEM info ---
        print(f"Ascii Path: {AsciiPath}")
        checkPRJ(AsciiPath)
        with rasterio.open(AsciiPath) as src:
            raster_bounds = box(*src.bounds)
            raster_crs = src.crs

        # --- 4. Align CRS ---
        shp = shp.to_crs(raster_crs)
        raster_gdf = gpd.GeoDataFrame({"geometry": [raster_bounds]}, crs=raster_crs)


        # --- 5. Keep only overlapping cities ---
        shp_in_raster = gpd.overlay(shp, raster_gdf, how="intersection")
        city = shp_in_raster[(shp_in_raster["NAME_1"] == cityList[0]) & (shp_in_raster["NAME_2"] == cityList[1])]


        # --- 7. Clip DEM to city boundary ---
        try:
            with rasterio.open(AsciiPath) as src:
                out_image, out_transform = mask(src, city.geometry, crop=True)
                out_meta = src.meta.copy()
        except ValueError:
            raise ValueError

        # --- 8. Update metadata for ASCII output ---
        out_meta.update({
            "driver": "AAIGrid",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # --- 9. Save as new ASCII ---
        ascii_out = rf"{base}\data\temp\{provinceName}\{cityName}\asc\{cityName}.asc"
        print(f"Saving Clipped DEM as: {ascii_out}")
        os.makedirs(os.path.dirname(ascii_out), exist_ok=True)

        with rasterio.open(ascii_out, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"Clipped ASCII DEM for {cityName}, saved as: {ascii_out}")
        return ascii_out
    



def calcRiverDist(demPath, riverPath, distX):
    """
    Returns the array of distance value from river network

    Parameters
    ----------
    demPath: str
        Path of the elevation data
    riverPath: str
        Path to the shp river network
    distX: array, np.array
        Array of cell size in x (Haversine or Euclidian)

    Returns
    -------
    out : Array
          Array object containing distance data.
      
    """

    # Load DEM
    with rasterio.open(demPath) as src:
        dem = src.read(1)
        profile = src.profile
        transform = src.transform
        crs = src.crs
        nodata = src.nodata



    # Load the clipped river shapefile
    rivers = gpd.read_file(riverPath).to_crs(crs)

    # Rasterize the river network
    # Create a binary mask where rivers = 1, non-rivers = 0
    river_mask = features.rasterize(
        ((geom, 1) for geom in rivers.geometry),
        out_shape=dem.shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    # Calculate Euclidean distance
    # Distance transform gives distance (in pixels) to nearest river pixel
    distance_pixels = distance_transform_edt(1 - river_mask)

    # Convert distance from pixels to map units (e.g., meters)
    distance_meters = distance_pixels * distX
    distance_meters[dem == -9999] = -9999 # Filter

    #with rasterio.open(f"{fileName}.asc", 'w', **profile) as dst:
        #dst.write(distance_meters.astype(np.float32), 1)
    return distance_meters


from pyproj import CRS
def checkPRJ(ascii_path, epsg_code=4326):
    """
    Creates a .prj file for an ASCII grid if it does not exist.
    """

    # Get .prj path
    prj_path = os.path.splitext(ascii_path)[0] + ".prj"

    # If already exists, do nothing
    if os.path.exists(prj_path):
        print("PRJ file already exists.")
        return prj_path

    # Create CRS from EPSG
    crs = CRS.from_epsg(epsg_code)

    # Convert to ESRI WKT format (important!)
    wkt = crs.to_wkt("WKT1_ESRI")

    # Write .prj file
    with open(prj_path, "w") as f:
        f.write(wkt)

    print(f"Created PRJ file: {prj_path}")
    return prj_path


def readHeaderBounds(path):
    with open(path, "r") as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip().split()
            header[line[0].lower()] = float(line[1])

    ncols = header["ncols"]
    nrows = header["nrows"]
    xll = header.get("xllcorner") or header.get("xllcenter")
    yll = header.get("yllcorner") or header.get("yllcenter")
    cellsize = header["cellsize"]

    xmin = xll
    ymin = yll
    xmax = xll + (ncols * cellsize)
    ymax = yll + (nrows * cellsize)

    return box(xmin, ymin, xmax, ymax)

main()