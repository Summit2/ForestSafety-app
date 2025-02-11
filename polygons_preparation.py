
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

def simplifyPolygons(mask, epsilon = 5):
    polygons = getPolygons(mask)
    for polygon in polygons:
        polygon = simplify_coords_vw(polygon,epsilon) #Висвалингам-Уатт
    return polygons

def polygonsToGeopolygons(polygons, bounds, affine_transformation):
    
    

    affine_matrix = np.array([
        [affine_transformation.a, affine_transformation.b, affine_transformation.c], 
        [affine_transformation.d, affine_transformation.e, affine_transformation.f], 
        [0, 0, 1]
    ])

    print(affine_matrix)

    geoPolygons = []
    for polygon in polygons:
        geoPoly = []
        for x, y in polygon:
            coords = np.array([y,x,1])

            geoCoords =  affine_matrix @ coords

            geoPoly.append(geoCoords.astype(int).tolist()[:-1])
        geoPolygons.append(geoPoly)
    return geoPolygons