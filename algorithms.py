import cv2 as cv
import numpy as np
from numpy import asarray
import math

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

def main():
    # load the image. The onl time when I rely on an external algorithm to process the image to working format for this this exercise
    im = cv.imread("lena512.pgm")
    im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)

    #nearest_Neighbor_Interpolation(im, (64,64))
    #linear_Method(im,(64,64),"X")
    #linear_Method(im,(64,64),"Y")
    #bilinear_Interpolation(im,(128,128))
    #grey_Level(im,8)
if __name__ == "__main__":
    main()