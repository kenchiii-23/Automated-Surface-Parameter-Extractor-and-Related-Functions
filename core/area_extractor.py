import geopandas as gpd
import numpy as np
from shapely.geometry import box
import rasterio
from rasterio import mask
from rasterio import features
from scipy.ndimage import distance_transform_edt


def  main():
    # --- 1. Load shapefile and define raster bounds ---
    base = r"C:\Users\KennethBaluyos (EPA)\Documents\04 - Manifesting RSTF\Projects\Project Reworked"
    shpPath = rf"{base}\data\raw\dependencies\NAMRIA_RiverNetwork\phl_rivl_250k_NAMRIA.shp"

    cutShpFromCity(shpPath, "Iligan City", "Rivers")


def cutShpFromCity(shpPath, cityName, shpName):
    # Load both shapefiles
    base = r"C:\Users\KennethBaluyos (EPA)\Documents\04 - Manifesting RSTF\Projects\Project Reworked"
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


def cutAsciFromCity(AsciiPath, cityName):
    base = r"C:\Users\KennethBaluyos (EPA)\Documents\04 - Manifesting RSTF\Projects\Project Reworked"
    shp_path = rf"{base}\data\raw\dependencies\cities\cities.shp"


    # --- 2. Load shapefile ---
    shp = gpd.read_file(shp_path)

    # --- 3. Load DEM info ---
    with rasterio.open(AsciiPath) as src:
        raster_bounds = box(*src.bounds)
        raster_crs = src.crs

    # --- 4. Align CRS ---
    shp = shp.to_crs(raster_crs)
    raster_gdf = gpd.GeoDataFrame({"geometry": [raster_bounds]}, crs=raster_crs)

    # --- 5. Keep only overlapping cities ---
    shp_in_raster = gpd.overlay(shp, raster_gdf, how="intersection")


    # --- 6. Select target city ---
    city = shp_in_raster[shp_in_raster["NAME_2"] == cityName]

    # --- 7. Clip DEM to city boundary ---
    with rasterio.open(AsciiPath) as src:
        out_image, out_transform = mask.mask(src, city.geometry, crop=True)
        out_meta = src.meta.copy()

    # --- 8. Update metadata for ASCII output ---
    out_meta.update({
        "driver": "AAIGrid",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # --- 9. Save as new ASCII ---
    ascii_out = rf"{base}\data\raw\{cityName.replace(' ', '_')}.asc"
    with rasterio.open(ascii_out, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Clipped ASCII DEM saved as: {ascii_out}")



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

main()