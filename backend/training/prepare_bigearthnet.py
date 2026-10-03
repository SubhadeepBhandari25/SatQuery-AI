from pathlib import Path

def prepare_bigearthnet_metadata():
    """Prepares class taxonomy and folder structure for BigEarthNet adaptation."""
    classes = [
        "Urban fabric", "Industrial or commercial units", "Arable land",
        "Permanent crops", "Pastures", "Complex cultivation patterns",
        "Broad-leaved forest", "Coniferous forest", "Mixed forest",
        "Natural grassland", "Moors and heathland", "Sparsely vegetated areas",
        "Inland wetlands", "Inland waters", "Marine waters"
    ]
    meta_path = Path(__file__).resolve().parent / "bigearthnet_classes.txt"
    meta_path.write_text("
".join(classes), encoding="utf-8")
    print(f"Prepared BigEarthNet class definitions: {len(classes)} classes in {meta_path.name}")

if __name__ == "__main__":
    prepare_bigearthnet_metadata()
