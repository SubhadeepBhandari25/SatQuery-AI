import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent

def create_synthetic_geotiff(
    filename: str,
    bands: int = 3,
    width: int = 512,
    height: int = 512,
    crs: str = "EPSG:32643", # UTM 43N (Delhi / ISRO region)
    origin_x: float = 718000.0,
    origin_y: float = 3165000.0,
    pixel_size: float = 10.0, # 10m spatial resolution
    scene_type: str = "optical"
) -> Path:
    """
    Creates an authentic GeoTIFF with genuine CRS, affine transform,
    bounds, and spectral characteristics (water, vegetation, built-up).
    """
    out_path = DATA_DIR / filename
    transform = from_origin(origin_x, origin_y, pixel_size, pixel_size)
    
    # Generate coherent remote sensing landscape
    y, x = np.ogrid[:height, :width]
    
    if scene_type == "optical":
        # Create landscape:
        # 1. Water body (curved river or lake in center)
        river = ((x - 200 - 40 * np.sin(y / 50.0)) ** 2 < 45**2)
        # 2. Vegetation (patches on right)
        veg = ((x > 300) & (y < 350)) | (((x - 100)**2 + (y - 400)**2) < 70**2)
        # 3. Built-up (dense grid pattern on top left)
        built = ((x < 220) & (y < 250) & ((x % 16 < 10) | (y % 16 < 10)))
        
        # Band 1: Red, Band 2: Green, Band 3: Blue, Band 4: NIR
        r = np.full((height, width), 160, dtype=np.uint8) # Soil baseline
        g = np.full((height, width), 140, dtype=np.uint8)
        b = np.full((height, width), 120, dtype=np.uint8)
        nir = np.full((height, width), 150, dtype=np.uint8)
        
        # Water: absorbs NIR, dark blue-green
        r[river] = 30
        g[river] = 75
        b[river] = 135
        nir[river] = 15
        
        # Vegetation: absorbs Red, high Green, very high NIR
        r[veg] = 45
        g[veg] = 160
        b[veg] = 50
        nir[veg] = 230
        
        # Built-up: high reflectance, sharp edges
        r[built] = 210
        g[built] = 205
        b[built] = 200
        nir[built] = 190

        if bands == 4:
            data = np.stack([r, g, b, nir], axis=0)
        else:
            data = np.stack([r, g, b], axis=0)

    elif scene_type == "sar":
        # SAR backscatter (microwave intensity):
        # Water: specular reflection away from sensor -> very low backscatter (-20 dB / ~25 intensity)
        # Built-up: dihedral double bounce -> very high backscatter (0 to 5 dB / ~230 intensity)
        # Vegetation: diffuse volume scattering -> medium backscatter (-10 dB / ~110 intensity)
        # Background: soil surface roughness (~70 intensity)
        sar_base = np.random.normal(70, 15, (height, width)).clip(20, 240).astype(np.uint8)
        
        river = ((x - 200 - 40 * np.sin(y / 50.0)) ** 2 < 45**2)
        veg = ((x > 300) & (y < 350)) | (((x - 100)**2 + (y - 400)**2) < 70**2)
        built = ((x < 220) & (y < 250) & ((x % 16 < 10) | (y % 16 < 10)))
        
        sar_base[river] = np.random.normal(25, 6, np.sum(river)).clip(5, 45)
        sar_base[built] = np.random.normal(225, 20, np.sum(built)).clip(180, 255)
        sar_base[veg] = np.random.normal(110, 18, np.sum(veg)).clip(60, 160)
        
        data = sar_base[np.newaxis, :, :]
        bands = 1

    elif scene_type == "optical_t2":
        # T2 for bi-temporal change: Urban development expanded into vegetation
        r = np.full((height, width), 160, dtype=np.uint8)
        g = np.full((height, width), 140, dtype=np.uint8)
        b = np.full((height, width), 120, dtype=np.uint8)
        
        river = ((x - 200 - 40 * np.sin(y / 50.0)) ** 2 < 45**2)
        r[river] = 30
        g[river] = 75
        b[river] = 135
        
        # Reduced vegetation due to new construction
        veg = ((x > 380) & (y < 350))
        r[veg] = 45
        g[veg] = 160
        b[veg] = 50
        
        # Expanded built-up (old built + new expansion in [300..380, y<250])
        built = ((x < 220) & (y < 250) & ((x % 16 < 10) | (y % 16 < 10)))
        new_dev = ((x >= 280) & (x < 370) & (y < 200) & ((x % 12 < 8) | (y % 12 < 8)))
        r[built | new_dev] = 215
        g[built | new_dev] = 210
        b[built | new_dev] = 205

        data = np.stack([r, g, b], axis=0)

    # Write GeoTIFF
    with rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
        
    print(f"Generated GeoTIFF: {out_path.name} (Bands: {bands}, CRS: {crs})")
    return out_path

def generate_all_samples():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 1. Optical 4-band GeoTIFF
    create_synthetic_geotiff("optical_sample_delhi.tif", bands=4, scene_type="optical")
    # 2. Co-registered SAR 1-band GeoTIFF
    create_synthetic_geotiff("sar_sample_delhi.tif", bands=1, scene_type="sar")
    # 3. Bi-temporal Pair (T1 and T2)
    create_synthetic_geotiff("bitemporal_t1.tif", bands=3, scene_type="optical")
    create_synthetic_geotiff("bitemporal_t2.tif", bands=3, scene_type="optical_t2")
    # 4. Standard non-georeferenced PNG for no-fake-coords testing
    img = Image.new("RGB", (256, 256), color=(70, 130, 180))
    png_path = DATA_DIR / "sample_non_geo.png"
    img.save(png_path)
    print(f"Generated Non-Geo image: {png_path.name}")

if __name__ == "__main__":
    generate_all_samples()
