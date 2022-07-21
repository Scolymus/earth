# -- Imports -- #
import base64
from fastapi import FastAPI, File, HTTPException
from io import BytesIO
from matplotlib import pyplot
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

app = FastAPI(
    title="EarthPulse",
    description="An API to solve the test for EarthPulse",
    version="1",
    contact="lpalacios@pm.me",
)

@app.get("/", tags=["root"])
def root():
    return {"message": "REST API developed for testings."}

@app.get(
    "/attributes", 
    name="attributes", 
    tags=["root"],
    summary="Returns image size (width and height), number of bands, \
            coordinate reference system and georeferenced bounding box"
    )
def attributes(image: bytes = File(...)):
    '''
    Read attributes from Sentinel image
    Args:
        image, bytes: Sentinel image file in bytes
    '''

    # Load image in memory
    with MemoryFile(image).open() as img:
        profile = img.profile
        geo_bb = img.bounds

    return {
        'width': profile['width'],
        'height': profile['height'],
        'num_bands': int(profile['count']),
        'coord_sys': str(profile['crs']),
        'geo_bb': str(geo_bb)
    }

@app.get(
    "/thumbnail", 
    name="thumbnail", 
    tags=["root"],
    summary="Returns a PNG thumbnail of the image"
    )
def thumbnail(image: bytes = File(...), bands: list[int] = [2,3,4], scale_down: int = 4, normalize_to_min_max: bool = True):
    '''
    Following documentation from Sentinel 2:
    https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/definitions
    True Colour Images are obtained from band 2, 3 and 4. Notice bands starts at 1!
    However, there is no information about what a "thumbnail" means.
    We could always want a preview for a given channel. Therefore, I'll allow this for the user
    At the same time, 'thumbnail' does not says anything about if it is a scaled version of the
    image, or not, and if it is, how much reduced it is. Therefore, I also allow to the user to do this

    Notice: bands must be 3 bands or 1. Otherwise it will take the first 3 if there are more than 1,
     or the first if there are 2
    Args:
        image, bytes: Sentinel image file in bytes
        bands, list: list of bands to show in preview. Either 1 or 3 bands in a list. Index starts in 1
        scale_down, int: Reduction factor for image
        normalize_to_min_max, bool: Images are 16-bit per channel. To reduce to 8-bit we can normalize
            per size of uint16 and then multiply per uint8 size (False) or normalize to uint8 setting
            the min value in image to 0 and max value to 255. Values in between are scaled linearly.
    '''

    if len(bands) > 3:
        bands = bands[:3]
    elif len(bands) == 2:
        bands = bands[0]
    elif len(bands) == 0:
        raise HTTPException(status_code=404, detail="Bands not found")

    print(bands, scale_down)

    # Load image in memory
    try:
        src = MemoryFile(image).open()
        h = src.height
        w = src.width
        # Becareful, resampling != 0 can cause negative values for pixels. Seen with resampling = 2
        data = src.read(indexes=bands, out=np.ndarray(shape=(len(bands), int(h/scale_down), int(w/scale_down))), resampling=0)
        src.close()
    except Exception as e:
        src.close()
        print(e)
        raise HTTPException(status_code=404, detail="Bands not found")

    return {
        'thumb': image_to_base64(data, bands, normalize_to_min_max)
    }

    '''
    ALTERNATIVE PROCESS

    # Load image in memory
    src = MemoryFile(image).open()

    # We can work with overviews, trying to find if they are present in the file
    # and if not creating them. For example, this code could do the trick.
    overviews = [src.overviews(i) for i in src.indexes]
    print(overviews)

    # Check if file contains overviews for the bands we want, and construct if not
    exist = True
    for o in bands:
        if overviews[o] == []:
            exist = False

    print(exist)

    if exist == False:
        src.build_overviews([scale_down], Resampling.average)    
        overviews = [src.overviews(i) for i in bands]

    # Take smallest overview, which is the maximum scale_down
    overview_num = overviews[bands[0]].index(max(overviews[bands[0]]))

    ...
    src.close()


    # The only problem about this procedure is that we need to make a
    memoryfile.write(data) 
    # because without it's a DatasetReader, not DatasetWriter, and DatasetReader does not have
    # build_overviews. But in an API, speed and cpu overloading is important...
    # so this would mean to check know which procedure is fastest
    '''

def image_to_base64(image, bands, normalize_to_min_max):

    # This step is crucial since rasterio orders image matrix differently from pillow
    thumb = np.moveaxis(image, [0, 1, 2], [2, 1, 0]).astype(np.float32)

    for axis in range(thumb.shape[2]):
        # Normalize to min-max values
        if normalize_to_min_max:
            mx = 1.*np.amax(thumb[:, :, axis])
            mi = 1.*np.amin(thumb[:, :, axis])

        # Normalize with respect maximum value possible in uint16
        elif normalize_to_min_max == False:
            mx = 65535
            mi = 0
        else:
            mx = 1.
            mi = -1.

        #print(mx, mi)
        thumb[:, :, axis] = (thumb[:, :, axis]-mi)/(mx-mi)


    #pyplot.imshow(thumb)
    #pyplot.show()

    # Select color mode
    if len(bands) == 1:
        thumb = Image.fromarray(np.uint8(thumb*255)[:,:,0], mode='L')
    else:
        thumb = Image.fromarray(np.uint8(thumb*255), mode='RGB')

    # Return it via base64
    buffered = BytesIO()
    thumb.save(buffered, format="PNG")
    thumb = base64.b64encode(buffered.getvalue())

    return thumb

@app.get(
    "/nvdi", 
    name="nvdi", 
    tags=["root"],
    summary="Computes an NDVI on the image and returns the result as a PNG"
    )
def nvdi(image: bytes = File(...), normalize_to_min_max: bool = False):
    '''
    NDVI = (NIR - RED) / (NIR + RED), where
    RED is B4, 664.5 nm
    NIR is B8, 835.1 nm
    
    Args:
        image, bytes: Sentinel image file in bytes
    Return:
        png image. 0(-1 ndvi)-255(+1 ndvi)
    '''
    bands = [4,8]
    try:
        src = MemoryFile(image).open()
        data = src.read(indexes=bands)
        src.close()
    except Exception as e:
        src.close()
        print(e)
        raise HTTPException(status_code=404, detail="Bands not found")

    ndvi = (data[1, :, :] - data[0, :, :]) / (data[1, :, :] + data[0, :, :])
    ndvi = ndvi[np.newaxis, :, :]
    # what to do with zeros?
    ndvi = np.where(ndvi == np.inf, -99, ndvi)

    return {
        'nvdi': image_to_base64(ndvi, [1], None)
    }