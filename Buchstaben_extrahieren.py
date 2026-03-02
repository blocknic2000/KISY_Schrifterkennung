import cv2
import numpy as np
import os

#User parameters
output_folder="letters"
image = cv2.imread("letters2.png")
number_rows=12
number_cols=8
margin=12 #Rand der weggeschnitten wird bei den einzelnen Buchstaben

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width, channels = image.shape
print(height)
print(width)

letter_width=width//number_cols
letter_height=height//number_rows

print(letter_width)
print(letter_height)

if not os.path.exists(output_folder):
        os.makedirs(output_folder)

for i in range(0,number_rows):
    for j in range(0,number_cols):
        x=j*letter_width
        y=i*letter_height
        if i==11:
             margin=4
        
        letter_img=gray[y+margin:y+letter_height-margin,x+margin:x+letter_width-margin]
        #if i==6 and j==8:
        #    cv2.imshow('letter_img', letter_img)
        # i in buchstaben umwandeln
        letter = chr(ord('A') + i+14)
        if not os.path.exists(output_folder+'/'+letter):
            os.makedirs(output_folder+'/'+letter)
        cv2.imwrite(output_folder+'/'+letter+"/"+str(j)+'.png', letter_img)


#cv2.imshow('BuchstabenRaster', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()