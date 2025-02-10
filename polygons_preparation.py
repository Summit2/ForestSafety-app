
from skimage import measure
from simplification.cutil import simplify_coords, simplify_coords_vw


def getPolygons(mask,level = 0.5):
    """
    mask -> two dims (example: (256,256))
    """
    # print(mask)
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
