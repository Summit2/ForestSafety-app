
from skimage import measure
from simplification.cutil import simplify_coords, simplify_coords_vw
import rasterio
import numpy as np


def getPolygons(mask,level = 0.5):
    """
    mask -> two dims (example: (256,256))
    """

    polygons = measure.find_contours(mask, level=level)
    for polygon in polygons:
        for i in range(polygon.shape[0]):
            polygon[i,1], polygon[i,0] =  polygon[i,0], polygon[i,1]
    return polygons

def simplifyPolygons(mask, epsilon=5):
    polygons = getPolygons(mask)
    simplified_polygons = []
    for polygon in polygons:
        simplified = simplify_coords_vw(polygon, epsilon)
        simplified_polygons.append(simplified)
    return simplified_polygons

from pyproj import Transformer

def polygonsToGeopolygons(polygons, transform, src_crs):
    """Преобразует полигоны в географические координаты (WGS84)"""
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    
    geoPolygons = []
    for polygon in polygons:
        geoPoly = []
        for y, x in polygon:  # y,x - координаты в пикселях
            # 1. Пиксели → UTM
            x_utm, y_utm = transform * (x, y)
            
            # 2. UTM → WGS84 (долгота, широта)
            lon, lat = transformer.transform(x_utm, y_utm)
            
            geoPoly.append([lon, lat])
        geoPolygons.append(geoPoly)
    return geoPolygons
