import cv2 as cv
import numpy as np
from numpy import asarray
import math
import matplotlib.pyplot as plt

def nearest_Neighbor_Interpolation(image,new_xy):
    #create a new blank image of size parameter 
    new_image = np.ones([new_xy[0],new_xy[1]], dtype=np.uint8)*255
    data = asarray(image)   #2d array

    #calculate ratio of x and y
    r_x, r_y = new_xy[0]/data.shape[0], new_xy[1]/data.shape[1]

    for x in range(new_xy[0]):
        for y in range(new_xy[1]):
            
            tx = math.floor(x/r_x)
            ty = math.floor(y/r_y)
            if new_image[x][y] == 255:
                new_image[x][y]=data[tx][ty]
    cv.imwrite("near.png", new_image)
    return new_image
    #cv.imshow('Example - Show image in window', new_image)
    #cv.waitKey(0) # waits until a key is pressed
    #cv.destroyAllWindows() # destroys the window showing image

def linear_Method(image,new_xy,choice):
    #create a new blank image of size parameter 
    new_image = np.ones([new_xy[0],new_xy[1]], dtype=np.uint8)*255
    data = asarray(image)
    
    #calculate ratio of x and y
    r_x, r_y = new_xy[0]/data.shape[0], new_xy[1]/data.shape[1]
    tx,ty = 0,0
    #y - y0 = ((y1 - y0)/(x1 - x0)) * (x - x0)
    #tx and ty values are rounded down to get whole number integers and not to get the overflow error when rounding up
    
    if choice == "X":
        counter = 0
        for x in range(new_xy[0]):
            if x <= (new_xy[0]/2-1):
                for y in range(new_xy[1]):
                    tx = math.floor((x/r_x) + ((data.shape[0]-(x/r_x))/(data.shape[1]-(y/r_y)))*((y/r_y)-data.shape[1]))         
                    ty = math.floor(y/r_y)
                    new_image[counter][y]=data[tx][ty]
                    new_image[counter+1][y]=data[tx][ty]
                counter = counter + 2                  

    if choice == "Y":
        for x in range(new_xy[0]):
            counter = 0
            for y in range(new_xy[1]):
                if y <= (new_xy[1]/2-1):
                    tx =  math.floor(x/r_x)                    
                    ty =  math.floor((y/r_y) + ((data.shape[1]-(y/r_y))/(data.shape[0]-(x/r_x)))*((x/r_x)-data.shape[0]))
                    new_image[x][counter]=data[tx][ty]
                    new_image[x][counter+1]=data[tx][ty]
                    counter = counter + 2
                   
  
    cv.imwrite(choice+"linear.png", new_image)
    return new_image
    #cv.imshow('Example - Show image in window', new_image)
    #cv.waitKey(0) # waits until a key is pressed
    #cv.destroyAllWindows() # destroys the window showing image

def bilinear_pixel(image,bi_X,bi_Y):
    modXi, modYi = int(bi_X),int(bi_Y)
    modXf, modYf = (bi_X - modXi),(bi_Y - modYi)

    modXLim = min(modXi+1,image.shape[1]-1)
    modYLim = min(modYi+1,image.shape[0]-1)

    bl = image[modYi, modXi]
    br = image[modYi, modXLim]
    tl = image[modYLim, modXi]
    tr = image[modYLim, modXLim]

    #Calculate interpolation
    b = modXf * br + (1 - modXf) * bl
    t = modXf * tr + (1 - modXf) * tl
    
    return int(modYf * t + (1 - modYf) * b+0.5)

def bilinear_Interpolation(image,new_xy):
    #create a new blank image of size parameter 
    new_image = np.ones([new_xy[0],new_xy[1]], dtype=np.uint8)*255
    data = asarray(image)

    #calculate ratio of x and y
    r_x, r_y = new_xy[0]/data.shape[0], new_xy[1]/data.shape[1]

    for x in range(new_xy[0]):
        for y in range(new_xy[1]):
            tx = x/r_x
            ty = y/r_y
            new_image[y][x]= bilinear_pixel(data,tx,ty)
    cv.imwrite("bilinear.png", new_image)
    return new_image
    #cv.imshow('Example - Show image in window', new_image)
    #cv.waitKey(0) # waits until a key is pressed
    #cv.destroyAllWindows() # destroys the window showing image

def grey_Level(image, g_scale):
    data = asarray(image)
    #np.set_printoptions(threshold=np.inf)
    #print(data)
    #3 bit = 8 levels => 256/8 = 32 per level ... assume image input is 3-bit
    # Bits 4-5 is the middle ground for the contrast of grey levels
    colorRange=[-128,-96,-64,-32,32,64,96,128]  #0-7
    scale = colorRange[g_scale-1]
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            temp = image[x][y] 
            if temp+scale <= 0:
                image[x][y] = 0
            elif temp+scale >= 255:
                image[x][y] = 255
            else:
                image[x][y] = temp+scale
    cv.imwrite("grey_Level"+str(g_scale)+".png", image)
    return image
    #cv.imshow('Example - Show image in window', image)
    #cv.waitKey(0) # waits until a key is pressed
    #cv.destroyAllWindows() # destroys the window showing image



#### Assignment #2  
def show_Img(title,img):
    cv.imshow(title, img)
    cv.waitKey(0) # waits until a key is pressed
    cv.destroyAllWindows() # destroys the window showing image   

def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    
    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1
    
    # return our final result
    return histogram

def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

def histogram_eq(image):
    data = asarray(image)
    flat = data.flatten()
    hist = get_histogram(flat, 256)
    cs = cumsum(hist)


    # re-normalize cumsum values to be between 0-255
    # numerator & denomenator
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    # re-normalize the cdf
    cs = nj / N
    #cast floating point to int8 values
    cs = cs.astype('uint8')
    # get the value from cumulative sum for every index in flat, and set that as img_new
    img_new = cs[flat]
    img_new = np.reshape(img_new, image.shape)

    cv.imwrite("histogram_global.png", img_new)
    return img_new
    show_Img("Local HE", img_new)

def histogram_eq_local(image, M):
    #locally with a 3x3, 5x5, 7x7 and 9x9 mask
   
    data = asarray(image)
    a,b = image.shape
    mask_3= [0,170,340,512]
    mask_5= [0,100,200,300,400,512]
    mask_7= [0,73,146,219,292,365,438,512]
    mask_9= [0,56,112,168,224,280,336,392,448,512]
    
    masks = [mask_3,mask_5,mask_7,mask_9]
    new_image = np.zeros([a, b])

    mask_len = len(masks[M])-1
    for indexX, x in enumerate(masks[M]):
        if indexX != mask_len:      
            for indexY, y in enumerate(masks[M]):
                if indexY != mask_len:
                    nextX = masks[M][indexX +1]
                    nextY = masks[M][indexY +1]
                    masked_tile = data[x:nextX,y:nextY]
                    new_image[x:nextX,y:nextY] = histogram_eq(masked_tile)
        
    cv.imwrite("histogram_local.png", new_image)
    return new_image
    show_Img("Local HE", new_image)

#default mask is assumed to be 3x3
def smooth_filter(image):
    #lowpass filtering
    data = asarray(image)
    a,b = image.shape
    
    mask = np.ones([3, 3], dtype = int)
    mask = mask / 9

    new_image = np.zeros([a, b])
    for i in range(1, a-1):
        for j in range(1, b-1):
            temp = data[i-1, j-1]*mask[0, 0]+data[i-1, j]*mask[0, 1]+data[i-1, j + 1]*mask[0, 2]+data[i, j-1]*mask[1, 0]+data[i, j]*mask[1, 1]+data[i, j + 1]*mask[1, 2]+data[i + 1, j-1]*mask[2, 0]+data[i + 1, j]*mask[2, 1]+data[i + 1, j + 1]*mask[2, 2]
            
            new_image[i, j]= temp

    new_image = new_image.astype(np.uint8)
    cv.imwrite("smoothfilter.png", new_image)
    return new_image
    show_Img("Smooth Filter", new_image)

def median_filter(image):
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    for i in range(1, a-1):
        for j in range(1, b-1):
            temp = [data[i-1, j-1],
                data[i-1, j],
                data[i-1, j + 1],
                data[i, j-1],
                data[i, j],
                data[i, j + 1],
                data[i + 1, j-1],
                data[i + 1, j],
                data[i + 1, j + 1]]
            
            temp = sorted(temp)
            new_image[i, j]= temp[4]
    
    new_image = new_image.astype(np.uint8)
    cv.imwrite("median_filter.png", new_image)
    return new_image
    show_Img("Median Filter", new_image)

def sharpen_Laplacian_filter(image):
    #high pass filter
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    mask = np.array([[1, 1, 1],
                    [1,-8, 1],
                    [1, 1, 1]])

    for i in range(1, a-1):
        for j in range(1, b-1):
            temp = data[i-1, j-1]*mask[0, 0]+data[i-1, j]*mask[0, 1]+data[i-1, j + 1]*mask[0, 2]+data[i, j-1]*mask[1, 0]+data[i, j]*mask[1, 1]+data[i, j + 1]*mask[1, 2]+data[i + 1, j-1]*mask[2, 0]+data[i + 1, j]*mask[2, 1]+data[i + 1, j + 1]*mask[2, 2]
            
            new_image[i, j]= data[i,j] - temp
            #sharp = orig - blurred  
    
    new_image = new_image.astype(np.uint8)
    cv.imwrite("sharplapacefilter.png", new_image)
    return new_image
    show_Img("SLF",new_image)

def high_boosting_filter(image, A):
    #A = boost factor
    
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    mask = np.array([[1, 1, 1],
                    [1,1, 1],
                    [1, 1, 1]])
    mask = mask/9

    for i in range(1, a-1):
        for j in range(1, b-1):
            temp = data[i-1, j-1]*mask[0, 0]+data[i-1, j]*mask[0, 1]+data[i-1, j + 1]*mask[0, 2]+data[i, j-1]*mask[1, 0]+data[i, j]*mask[1, 1]+data[i, j + 1]*mask[1, 2]+data[i + 1, j-1]*mask[2, 0]+data[i + 1, j]*mask[2, 1]+data[i + 1, j + 1]*mask[2, 2]
            
            blur = A *( data[i,j] - temp)

            new_image[i, j]= data[i,j] + blur
            
    
    new_image = new_image.astype(np.uint8)
    cv.imwrite("HBFilter.png", new_image)
    return new_image
    #show_Img("HBF",new_image) 

def bit_plane_slice(image,bit):
    data = asarray(image)
    a,b = image.shape
    lst = []
    for i in range(0, a):
        for j in range(0, b):
            lst.append(np.binary_repr(data[i][j] ,width=8))
    
    #Seperating bits into different bit arrays
    eight_bit = (np.array([int(i[0]) for i in lst],dtype = np.uint8) * 128).reshape(a,b)
    seven_bit = (np.array([int(i[1]) for i in lst],dtype = np.uint8) * 64).reshape(a,b)
    six_bit = (np.array([int(i[2]) for i in lst],dtype = np.uint8) * 32).reshape(a,b)
    five_bit = (np.array([int(i[3]) for i in lst],dtype = np.uint8) * 16).reshape(a,b)
    four_bit = (np.array([int(i[4]) for i in lst],dtype = np.uint8) * 8).reshape(a,b)
    three_bit = (np.array([int(i[5]) for i in lst],dtype = np.uint8) * 4).reshape(a,b)
    two_bit = (np.array([int(i[6]) for i in lst],dtype = np.uint8) * 2).reshape(a,b)
    one_bit = (np.array([int(i[7]) for i in lst],dtype = np.uint8) * 1).reshape(a,b)


    #Exporting bit-wise image strip from high to low
    high_bits = cv.hconcat([eight_bit,seven_bit,six_bit,five_bit])
    low_bits =cv.hconcat([four_bit,three_bit,two_bit,one_bit])
    strip = cv.vconcat([high_bits,low_bits])
    cv.imwrite("BitSlicingOverall.png", strip)
  
 
    if bit == "HIGH":
        new_image = eight_bit + seven_bit + six_bit + five_bit
    else:
        new_image = four_bit + three_bit + two_bit + one_bit
    cv.imwrite("BitSlicing.png", new_image)
    return new_image
    show_Img("Bit Splicing - " + bit,new_image)
    show_Img("Recombined",eight_bit + seven_bit + six_bit + five_bit+four_bit + three_bit + two_bit + one_bit)



#~~~~~~~~ Assignment 3 ~~~~~~~~~~~~~~
def mask_gen(image,mask,x,y):
    temp_arr = []
    #print(str(mask))
    #print("Image pixel: "+ str(x) +"," + str(y) +": " + str(image[x][y]))
    if mask >=3:    #mask 3x3
        temp_arr.append(image[x-1][y-1])
        temp_arr.append(image[x-1][y])
        temp_arr.append(image[x-1][y+1])

        temp_arr.append(image[x][y-1])
        temp_arr.append(image[x][y])
        temp_arr.append(image[x][y+1])

        temp_arr.append(image[x+1][y-1])
        temp_arr.append(image[x+1][y])
        temp_arr.append(image[x+1][y+1])
        
    if mask >=5:    #mask 5x5 add ontop of 3x3 mask
        temp_arr.append(image[x-2][y-2])
        temp_arr.append(image[x-2][y-1])
        temp_arr.append(image[x-2][y])
        temp_arr.append(image[x-2][y+1])
        temp_arr.append(image[x-2][y+2])

        temp_arr.append(image[x-1][y-2])
        temp_arr.append(image[x-1][y+2])

        temp_arr.append(image[x][y-2])
        temp_arr.append(image[x][y+2])

        temp_arr.append(image[x+1][y-2])
        temp_arr.append(image[x+1][y+2])

        temp_arr.append(image[x+2][y-2])
        temp_arr.append(image[x+2][y-1])
        temp_arr.append(image[x+2][y])
        temp_arr.append(image[x+2][y+1])
        temp_arr.append(image[x+2][y+2])
      
    if mask >=7:
        temp_arr.append(image[x-3][y-3])
        temp_arr.append(image[x-3][y-2])
        temp_arr.append(image[x-3][y-1])
        temp_arr.append(image[x-3][y])
        temp_arr.append(image[x-3][y+1])
        temp_arr.append(image[x-3][y+2])
        temp_arr.append(image[x-3][y+3])

        temp_arr.append(image[x-2][y-3])
        temp_arr.append(image[x-2][y+3])
        
        temp_arr.append(image[x-1][y-3])
        temp_arr.append(image[x-1][y+3])

        temp_arr.append(image[x][y-3])
        temp_arr.append(image[x][y+3])

        temp_arr.append(image[x+1][y-3])
        temp_arr.append(image[x+1][y+3])

        temp_arr.append(image[x+2][y-3])
        temp_arr.append(image[x+2][y+3])

        temp_arr.append(image[x+3][y-3])
        temp_arr.append(image[x+3][y-2])
        temp_arr.append(image[x+3][y-1])
        temp_arr.append(image[x+3][y])
        temp_arr.append(image[x+3][y+1])
        temp_arr.append(image[x+3][y+2])
        temp_arr.append(image[x+3][y+3])

    if mask >=9:   
        temp_arr.append(image[x-4][y-4])
        temp_arr.append(image[x-4][y-3])
        temp_arr.append(image[x-4][y-2])
        temp_arr.append(image[x-4][y-1])
        temp_arr.append(image[x-4][y])
        temp_arr.append(image[x-4][y+1])
        temp_arr.append(image[x-4][y+2])
        temp_arr.append(image[x-4][y+3])
        temp_arr.append(image[x-4][y+4])

        temp_arr.append(image[x-3][y-4])
        temp_arr.append(image[x-3][y+4])

        temp_arr.append(image[x-2][y-4])
        temp_arr.append(image[x-2][y+4])
        
        temp_arr.append(image[x-1][y-4])
        temp_arr.append(image[x-1][y+4])

        temp_arr.append(image[x][y-4])
        temp_arr.append(image[x][y+4])

        temp_arr.append(image[x+1][y-4])
        temp_arr.append(image[x+1][y+4])

        temp_arr.append(image[x+2][y-4])
        temp_arr.append(image[x+2][y+4])

        temp_arr.append(image[x+3][y-4])
        temp_arr.append(image[x+3][y+4])

        temp_arr.append(image[x+4][y-4])
        temp_arr.append(image[x+4][y-3])
        temp_arr.append(image[x+4][y-2])
        temp_arr.append(image[x+4][y-1])
        temp_arr.append(image[x+4][y])
        temp_arr.append(image[x+4][y+1])
        temp_arr.append(image[x+4][y+2])
        temp_arr.append(image[x+4][y+3])
        temp_arr.append(image[x+4][y+4])
    #print("Image pixel array: "+ str(x) +"," + str(y) +": " + str(len(temp_arr)))
    return temp_arr

def QuickSort(arr):
    less = []
    equal = []
    greater = []

    if len(arr) > 1:
        pivot = arr[0]
        for x in arr:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)

        return QuickSort(less)+equal+QuickSort(greater)  
    else:  
        return arr

def Arithmetic_mean_filter(image,mask):
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    msk_sz = (mask-1)/2 #1,2,3,4 -> 3*3,5*5,7*7,9*9
    msk_sz = int(msk_sz)
    for i in range(msk_sz,a-msk_sz):
        for j in range(msk_sz,b-msk_sz):
            arr=mask_gen(data,mask, i,j)  
          
            total = 0
            for k in range(len(arr)):
                total += arr[k]
            total=int(total/(mask*mask))
          
            new_image[i][j]=total
    cv.imwrite("Arithmean.png", new_image)
    return new_image

def Geometric_mean_filter(image,mask):
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    msk_sz = (mask-1)/2 #1,2,3,4 -> 3*3,5*5,7*7,9*9
    msk_sz = int(msk_sz)
    
    for i in range(msk_sz,a-msk_sz):
        for j in range(msk_sz,b-msk_sz):
            arr=mask_gen(data,mask, i,j)
            #print(str(arr))
            #mult = np.prod(arr) 
            #GEO_res = mult ** (1.0/(mask*mask))
         
            
            #G= (1*2*3 ..*n)^(1.0/n)
            G = np.log(arr)
            G = np.exp(G.mean())
            #print(str(int(GEO_res)))
            new_image[i][j]=int(G)
    cv.imwrite("Geometric.png", new_image)
    return new_image

def Harmonic_mean_filter(image,mask):
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    msk_sz = (mask-1)/2 #1,2,3,4 -> 3*3,5*5,7*7,9*9
    msk_sz = int(msk_sz)
    
    for i in range(msk_sz,a-msk_sz):
        for j in range(msk_sz,b-msk_sz):
            arr=mask_gen(data,mask, i,j)
            har = 0
            for k in range(len(arr)): 
                har += (1/arr[k]) 
            Harmonic = ((mask*mask)/har)
            new_image[i][j] =int(Harmonic)
    cv.imwrite("Harmonic.png", new_image)
    return new_image

def Contraharmonic_mean_filter(image,mask, Q):
    if Q == -1:
        return Harmonic_mean_filter(image,mask)
    if Q == 0:
        return Arithmetic_mean_filter(image,mask)
    
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    msk_sz = (mask-1)/2 #1,2,3,4 -> 3*3,5*5,7*7,9*9
    msk_sz = int(msk_sz)
    
    for i in range(msk_sz,a-msk_sz):
        for j in range(msk_sz,b-msk_sz):
            arr=mask_gen(data,mask, i,j)
            num = 0
            dem = 0
            for k in range(len(arr)):
                num += arr[k] ** (Q+1)
                dem += arr[k] ** (Q)
            contraHar = int(num/dem)
            new_image[i][j] = contraHar
    cv.imwrite("ContraHarmonic.png", new_image)
    return new_image

def Max_filter(image,mask):
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    msk_sz = (mask-1)/2 #1,2,3,4 -> 3*3,5*5,7*7,9*9
    msk_sz = int(msk_sz)
    
    for i in range(msk_sz,a-msk_sz):
        for j in range(msk_sz,b-msk_sz):
            arr=mask_gen(data,mask, i,j)
            new_image[i][j]=max(arr)
    cv.imwrite("Max.png", new_image)
    return new_image

def Min_filter(image,mask):
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    msk_sz = (mask-1)/2 #1,2,3,4 -> 3*3,5*5,7*7,9*9
    msk_sz = int(msk_sz)
    
    for i in range(msk_sz,a-msk_sz):
        for j in range(msk_sz,b-msk_sz):
            arr=mask_gen(data,mask, i,j)
            new_image[i][j]=min(arr)
    cv.imwrite("Min.png", new_image)
    return new_image

def Midpoint_filter(image,mask):
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    msk_sz = (mask-1)/2 #1,2,3,4 -> 3*3,5*5,7*7,9*9
    msk_sz = int(msk_sz)
    
    for i in range(msk_sz,a-msk_sz):
        for j in range(msk_sz,b-msk_sz):
            arr=mask_gen(data,mask, i,j)
            minA = min(arr)
            maxA = max(arr)
            new_image[i][j]=(minA+maxA)/2
    cv.imwrite("Midpoint.png", new_image)
    return new_image

def Alpha_trimmed_mean_filter(image,mask,alpha):
    data = asarray(image)
    a,b = image.shape
    new_image = np.zeros([a, b])

    msk_sz = (mask-1)/2 #1,2,3,4 -> 3*3,5*5,7*7,9*9
    msk_sz = int(msk_sz)
    
    for i in range(msk_sz,a-msk_sz):
        for j in range(msk_sz,b-msk_sz):
            arr=mask_gen(data,mask, i,j)
            arr = QuickSort(arr)    #order elements
                    
            # Calculate the trim coefficient
            d = int((mask*mask)*(alpha)/2)
            #print(str(b))
            if (d != 0):
                arr = arr[d:len(arr)-d]
            #print(arr)
            total = 0
            for k in range(len(arr)):
                total += arr[k]
            #print(total)       
            total=int(total/len(arr))
            #print(str(len(arr)))
         
            #print(total)
            new_image[i][j]=total
    cv.imwrite("Alpha.png", new_image)
    return new_image

def main():
    # load the image. The only time when I rely on an external algorithm to process the image to working format for this this exercise
    #im = cv.imread("lena512.pgm")
    im = cv.imread("Fig0507g.tif")
    im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    #show_Img("Orignal", im)
    
    #nearest_Neighbor_Interpolation(im, (64,64))
    #linear_Method(im,(64,64),"X")
    #linear_Method(im,(64,64),"Y")
    #bilinear_Interpolation(im,(128,128))
    #grey_Level(im,8)
    #histogram_eq(im)
    #histogram_eq_local(im,3)
    #smooth_filter(im)
    #median_filter(im)
    #sharpen_Laplacian_filter(im)
    #high_boosting_filter(im,50)
    #bit_plane_slice(im,"HIGH")
    #Arithmetic_mean_filter(im,9)
    Geometric_mean_filter(im,9)
    #Harmonic_mean_filter(im,3)
    #Contraharmonic_mean_filter(im,3,1.5)
    #Max_filter(im,3)
    #Min_filter(im,3)
    #Midpoint_filter(im,3)
    #Alpha_trimmed_mean_filter(im,3,0.2)
if __name__ == "__main__":
    main()