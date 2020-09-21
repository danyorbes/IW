import cv2

import imutils
import numpy as np
import pytesseract
import webbrowser
from PIL import Image

cars_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

cap = cv2.VideoCapture(0)


def detect_cars(frame):

    cars = cars_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in cars:
        plate = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
        cv2.imshow('car',plate)
        cv2.imwrite("carplate.jpg", plate)
        
    return frame

def placa():
    #img = cv2.VideoCapture('carplate.jpeg',cv2.IMREAD_COLOR)
    img = cv2.imread('carplate.jpg',cv2.IMREAD_COLOR)

    img = cv2.resize(img, (620,480) )

    placaencontrada=False


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convierte a escala de grises
    gray = cv2.bilateralFilter(gray, 11, 17, 17) #Desenfoque para reducir el ruido ***************
    edged = cv2.Canny(gray, 30, 200) #Realizar detección de bordes*********************


    # encontrar contornos en la imagen de bordes, mantener solo el más grande
    # ones, e inicializa nuestro contorno de pantalla
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    # bucle sobre nuestros contornos
    for c in cnts:
        
        # aproximar el contorno
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    
        # si nuestro contorno aproximado tiene cuatro puntos, entonces
        # podemos suponer que hemos encontrado nuestra pantalla
        if len(approx) == 4:
            screenCnt = approx
            break



    if screenCnt is None:
        detected = 0
        print ("No contour detected")
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

    # Enmascarar la parte que no sea la placa de matrícula
    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)


    # recortar
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    #FILTROS
    ret,thresh1 = cv2.threshold(Cropped,127,255,cv2.THRESH_BINARY)
    ret,thresh3 = cv2.threshold(thresh1,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(Cropped,127,255,cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(Cropped,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(Cropped,(5,5),0)
    ret, thresh6 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #Eliminacion de ruido
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel, iterations = 2)
    # Encuentra el área del fondo
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Encuentra el área del primer
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)



    #Lee la matrícula

    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    text2 = pytesseract.image_to_string(thresh1, config='--psm 11')
    text3 = pytesseract.image_to_string(thresh3, config='--psm 11')
    text4 = pytesseract.image_to_string(thresh4, config='--psm 11')
    text5 = pytesseract.image_to_string(thresh5, config='--psm 11')
    text6 = pytesseract.image_to_string(thresh6, config='--psm 11')


    if text [0] == "{":
        text = text.replace("{", "I")
        
    if text [0] == "/":
        text = text.replace("/", "I")

    if text [0] == "1":
        text = text.replace("1", "I")

        

    if text2 [0] == "{":
        text2 = text2.replace("{", "I")
        
    if text2 [0] == "/":
        text2 = text2.replace("/", "I")

    if text2 [0] == "1":
        text2 = text2.replace("1", "I")
        

    if text3 [0] == "{":
        text3 = text3.replace("{", "I")
        
    if text3 [0] == "/":
        text3 = text3.replace("/", "I")

    if text3 [0] == "1":
        text3 = text3.replace("1", "I")
        

    if text4 [0] == "{":
        text4 = text4.replace("{", "I")
        
    if text4 [0] == "/":
        text4 = text4.replace("/", "I")

    if text4 [0] == "1":
        text4 = text4.replace("1", "I")
        

    if text5 [0] == "{":
        text5 = text5.replace("{", "I")
        
    if text5 [0] == "/":
        text5 = text5.replace("/", "I")

    if text5 [0] == "1":
        text5 = text5.replace("1", "I")


    if text6 [0] == "{":
        text6 = text6.replace("{", "I")
        
    if text6 [0] == "/":
        text6 = text6.replace("/", "I")

    if text6 [0] == "1":
        text6 = text6.replace("1", "I")


        
    #Longitud de texto por cada filtro
    longitud1 = len(text)
    longitud2 = len(text2)
    longitud3 = len(text3)
    longitud4 = len(text4)
    longitud5 = len(text5)
    longitud6 = len(text6)



    #Procedemos a normalizar y validar cada uno de los filtros
    print("Detected Number is:",text)

    print("1ER FILTRO")

    if longitud1 > 5:
        comp11 = text [0].isalpha()
        comp12 = text [1].isalpha()
        comp13 = text [2].isalpha()

        comp14 = text [4].isdigit() 
        comp15 = text [5].isdigit()
        comp16 = text [6].isdigit() 

        if comp11 == True:
            print("1 parametro valido")
            if comp12 == True:
                print("2 parametro valido")
                if comp13 == True:
                    print("3 parametro valido")
                    if text [3] == "-":
                        print("Gion conseguido")
                        if comp14 == True:
                            print("4 parametro valido")
                            if comp15 == True:
                                print("5 parametro valido")
                                if comp16 == True:
                                    print("6 parametro valido")
                                    valides1 = True
                                else:
                                    print("6 parametro no valido")
                                    valides1 = False
                            else:
                                print("5 parametro no valido")
                                valides1 = False
                        else:
                            print("4 parametro no valido")
                            valides1 = False
                    else:
                        print("no es placa nacional")
                        valides1 = False
                else:
                    print("3 parametro no valido")
                    valides1 = False
            else:
                print("2 parametro no valido")
                valides1 = False
        else:
            print("1 parametro no valido")
            valides1 = False
    else:
        print("el primer filtro no es valido")
        valides1 = False


    print("Detected Number2 is:",text2)

    print("2DO FILTRO")
    if longitud2 > 5:
        comp21 = text2 [0].isalpha()
        comp22 = text2 [1].isalpha()
        comp23 = text2 [2].isalpha()

        comp24 = text2 [4].isdigit() 
        comp25 = text2 [5].isdigit() 
        comp26 = text2 [6].isdigit() 

        if comp21 == True:
            print("1 parametro valido")
            if comp22 == True:
                print("2 parametro valido")
                if comp23 == True:
                    print("3 parametro valido")
                    if text2 [3] == "-":
                        print("Gion conseguido")
                        if comp24 == True:
                            print("4 parametro valido")
                            if comp25 == True:
                                print("5 parametro valido")
                                if comp26 == True:
                                    print("6 parametro valido")
                                    valides2 = True
                                else:
                                    print("6 parametro no valido")
                                    valides2 = False
                            else:
                                print("5 parametro no valido")
                                valides2 = False
                        else:
                            print("4 parametro no valido")
                            valides2 = False
                    else:
                        print("no es placa nacional")
                        valides2 = False
                else:
                    print("3 parametro no valido")
                    valides2 = False
            else:
                print("2 parametro no valido")
                valides2 = False
        else:
            print("1 parametro no valido")
            valides2 = False
    else:
        print("el segundo filtro no es valido")
        valides2 = False


    print("Detected Number3 is:",text3)

    print("3RO FILTRO")
    if longitud3 > 5:
        comp31 = text3 [0].isalpha()
        comp32 = text3 [1].isalpha()
        comp33 = text3 [2].isalpha()

        comp34 = text3 [4].isdigit() 
        comp35 = text3 [5].isdigit() 
        comp36 = text3 [6].isdigit() 

        if comp31 == True:
            print("1 parametro valido")
            if comp32 == True:
                print("2 parametro valido")
                if comp33 == True:
                    print("3 parametro valido")
                    if text3 [3] == "-":
                        print("Gion conseguido")
                        if comp34 == True:
                            print("4 parametro valido")
                            if comp35 == True:
                                print("5 parametro valido")
                                if comp36 == True:
                                    print("6 parametro valido")
                                    valides3 = True
                                else:
                                    print("6 parametro no valido")
                                    valides3 = False
                            else:
                                print("5 parametro no valido")
                                valides3 = False
                        else:
                            print("4 parametro no valido")
                            valides3 = False
                    else:
                        print("no es placa nacional")
                        valides3 = False
                else:
                    print("3 parametro no valido")
                    valides3 = False
            else:
                print("2 parametro no valido")
                valides3 = False
        else:
            print("1 parametro no valido")
            valides3 = False
    else:
        print("el tercer filtro no es valido")
        valides3 = False


    print("Detected Number4 is:",text4)
    print("4TO FILTRO")
    if longitud4 > 5:
        comp41 = text4 [0].isalpha()
        comp42 = text4 [1].isalpha()
        comp43 = text4 [2].isalpha()

        comp44 = text4 [4].isdigit() 
        comp45 = text4 [5].isdigit() 
        comp46 = text4 [6].isdigit() 

        if comp41 == True:
            print("1 parametro valido")
            if comp42 == True:
                print("2 parametro valido")
                if comp43 == True:
                    print("3 parametro valido")
                    if text4 [3] == "-":
                        print("Gion conseguido")
                        if comp44 == True:
                            print("4 parametro valido")
                            if comp45 == True:
                                print("5 parametro valido")
                                if comp46 == True:
                                    print("6 parametro valido")
                                    valides4 = True
                                else:
                                    print("6 parametro no valido")
                                    valides4 = False
                            else:
                                print("5 parametro no valido")
                                valides4 = False
                        else:
                            print("4 parametro no valido")
                            valides4 = False
                    else:
                        print("no es placa nacional")
                        valides4 = False
                else:
                    print("3 parametro no valido")
                    valides4 = False
            else:
                print("2 parametro no valido")
                valides4 = False
        else:
            print("1 parametro no valido")
            valides4 = False
    else:
        print("el cuarto filtro no es valido")
        valides4 = False


    print("Detected Number5 is:",text5)
    print("5TO FILTRO")
    if longitud5 > 5:
        comp51 = text5 [0].isalpha()
        comp52 = text5 [1].isalpha()
        comp53 = text5 [2].isalpha()

        comp54 = text5 [4].isdigit() 
        comp55 = text5 [5].isdigit() 
        comp56 = text5 [6].isdigit() 

        if comp51 == True:
            print("1 parametro valido")
            if comp52 == True:
                print("2 parametro valido")
                if comp53 == True:
                    print("3 parametro valido")
                    if text5 [3] == "-":
                        print("Gion conseguido")
                        if comp54 == True:
                            print("4 parametro valido")
                            if comp55 == True:
                                print("5 parametro valido")
                                if comp56 == True:
                                    print("6 parametro valido")
                                    valides5 = True
                                else:
                                    print("6 parametro no valido")
                                    valides5 = False
                            else:
                                print("5 parametro no valido")
                                valides5 = False
                        else:
                            print("4 parametro no valido")
                            valides5 = False
                    else:
                        print("no es placa nacional")
                        valides5 = False
                else:
                    print("3 parametro no valido")
                    valides5 = False
            else:
                print("2 parametro no valido")
                valides5 = False
        else:
            print("1 parametro no valido")
            valides5 = False
    else:
        print("el quinto filtro no es valido")
        valides5 = False

    print("Detected Number6 is:",text6)
    print("6TO FILTRO")
    if longitud6 > 6:
        comp61 = text6 [0].isalpha()
        comp62 = text6 [1].isalpha()
        comp63 = text6 [2].isalpha()

        comp64 = text6 [4].isdigit() 
        comp65 = text6 [5].isdigit() 
        comp66 = text6 [6].isdigit() 

        if comp61 == True:
            print("1 parametro valido")
            if comp62 == True:
                print("2 parametro valido")
                if comp63 == True:
                    print("3 parametro valido")
                    if text6 [3] == "-":
                        print("Gion conseguido")
                        if comp64 == True:
                            print("4 parametro valido")
                            if comp65 == True:
                                print("5 parametro valido")
                                if comp66 == True:
                                    print("6 parametro valido")
                                    valides6 = True
                                else:
                                    print("6 parametro no valido")
                                    valides6 = False
                            else:
                                print("5 parametro no valido")
                                valides6 = False
                        else:
                            print("4 parametro no valido")
                            valides6 = False
                    else:
                        print("no es placa nacional")
                        valides6 = False
                else:
                    print("3 parametro no valido")
                    valides6 = False
            else:
                print("2 parametro no valido")
                valides6 = False
        else:
            print("1 parametro no valido")
            valides6 = False
    else:
        print("el sexto filtro no es valido")
        valides6 = False


    



    if valides1 == True:
        
        comprobacion()
        #resultado1
        if coincidencia > 1:
            print("El resultado con un porcentaje de:",(coincidencia+1*100/6))
            print("es:",text)
            valides2 = False
            valides3 = False
            valides4 = False
            valides5 = False
            valides6 = False
            placaencontrada = True
            recordTuple = (text)

    if valides2 == True:
        
        comprobacion2()
        #resultado2
        
        if coincidencia2 >= 1:
            print("El resultado con un porcentaje de:",(coincidencia2+1)*100/6)
            print("es:",text2)
            valides3 = False
            valides4 = False
            valides5 = False
            valides6 = False
            placaencontrada = True
            recordTuple = (text2)
        
        

    if valides3 == True:
        
        comprobacion3()
        #resultado3
        
        if coincidencia3 >= 1:
            print("El resultado con un porcentaje de:",(coincidencia3+1)*100/6)
            print("es:",text3)
            valides3 = False
            valides4 = False
            valides5 = False
            valides6 = False
            placaencontrada = True
            recordTuple = (text3)

    if valides4 == True:
        
        comprobacion4()
        #resultado4
        
        if coincidencia4 >= 1:
            print("El resultado con un porcentaje de:",(coincidencia4+1)*100/6)
            print("es:",text4)
            
            valides4 = False
            valides5 = False
            valides6 = False
            placaencontrada = True
            recordTuple = (text4)

    if valides5 == True:
        
        comprobacion5()
        #resultado4
        
        if coincidencia5 >= 1:
            print("El resultado con un porcentaje de:",(coincidencia5+1)*100/6)
            print("es:",text5)
            
            valides6 = False
            placaencontrada = True
            recordTuple = (text5)

    if valides6 == True:
        
        comprobacion6()
        #resultado4
        
        if coincidencia6 >= 1:
            print("El resultado con un porcentaje de:",(coincidencia6+1)*100/6)
            print("es:",text6)
            placaencontrada = True
            recordTuple = (text6)

    if valides6 == True:
            print("NECESITAMOS OTRA FOTO")
            

    #print("l1:",text4 [0])
    #print("l2:",text4 [1])
    #print("l3:",text4 [2])
    #print("l4:",text4 [3])
    #print("l5:",text4 [4])
    #print("l6:",text4 [5])
    #print("l7:",text4 [6])
    #print("l8:",text4 [7])

    if placaencontrada == True:
        print("fotografia si")
        webbrowser.open_new("https://parqueaderoapp.000webhostapp.com/parqueadero/agregar.php?placa="+ recordTuple)
    else:
        print("fotografia no")
        
    cv2.imshow('image',img)
    cv2.imshow('Cropped',Cropped)
    cv2.imshow('img_binary',thresh1)
    cv2.imshow('img_trunc',thresh3)
    cv2.imshow('tozero',thresh4)
    cv2.imshow('binaryinv',thresh5)
    cv2.imshow('Otsus Binarization',thresh6)
    print("tesis")
    return


#Comparamos los validos
    def comprobacion():
        global coincidencia
        coincidencia =0
        busca = text.find(text2)
        if busca == 0:
            coincidencia = coincidencia+1
        busca = text.find(text3)
        if busca == 0:
            coincidencia = coincidencia+1
        busca = text.find(text4)
        if busca == 0:
            coincidencia = coincidencia+1
        busca = text.find(text5)
        if busca == 0:
            coincidencia = coincidencia+1
        busca = text.find(text6)
        if busca == 0:
            coincidencia = coincidencia+1
        
            
        
        return
    def comprobacion2():
        global coincidencia2 
        coincidencia2 =0 
        busca = text2.find(text)
        if busca == 0:
            coincidencia2 = coincidencia2+1
        busca = text2.find(text3)
        if busca == 0:
            coincidencia2 = coincidencia2+1
        busca = text2.find(text4)
        if busca == 0:
            coincidencia2 = coincidencia2+1
        busca = text2.find(text5)
        if busca == 0:
            coincidencia2 = coincidencia2+1
        busca = text2.find(text6)
        if busca == 0:
            coincidencia2 = coincidencia2+1
        return

    def comprobacion3():
        global coincidencia3 
        coincidencia3=0
        busca = text3.find(text)
        if busca == 0:
            coincidencia3 = coincidencia3+1
        busca = text3.find(text2)
        if busca == 0:
            coincidencia3 = coincidencia3+1
        busca = text3.find(text4)
        if busca == 0:
            coincidencia3 = coincidencia3+1
        busca = text3.find(text5)
        if busca == 0:
            coincidencia3 = coincidencia3+1
        busca = text3.find(text6)
        if busca == 0:
            coincidencia3 = coincidencia3+1
        print(coincidencia3)
        return

    def comprobacion4():
        global coincidencia4
        coincidencia4=0
        busca = text4.find(text)
        if busca == 0:
            coincidencia4 = coincidencia4+1
        busca = text4.find(text2)
        if busca == 0:
            coincidencia4 = coincidencia4+1
        busca = text4.find(text3)
        if busca == 0:
            coincidencia4 = coincidencia4+1
        busca = text4.find(text5)
        if busca == 0:
            coincidencia4 = coincidencia4+1
        busca = text4.find(text6)
        if busca == 0:
            coincidencia4 = coincidencia4+1
        
        return
        
    def comprobacion5():
        global coincidencia5
        coincidencia5=0
        busca = text5.find(text)
        if busca == 0:
            coincidencia5 = coincidencia5+1
        busca = text5.find(text2)
        if busca == 0:
            coincidencia5 = coincidencia5+1
        busca = text5.find(text3)
        if busca == 0:
            coincidencia5 = coincidencia5+1
        busca = text5.find(text4)
        if busca == 0:
            coincidencia5 = coincidencia5+1
        busca = text5.find(text6)
        if busca == 0:
            coincidencia5 = coincidencia5+1
        
        return

    def comprobacion6():
        global coincidencia6
        coincidencia6=0
        busca = text6.find(text)
        if busca == 0:
            coincidencia6 = coincidencia6+1
        busca = text6.find(text2)
        if busca == 0:
            coincidencia6 = coincidencia6+1
        busca = text6.find(text3)
        if busca == 0:
            coincidencia6 = coincidencia6+1
        busca = text6.find(text4)
        if busca == 0:
            coincidencia6 = coincidencia6+1
        busca = text6.find(text5)
        if busca == 0:
            coincidencia6 = coincidencia6+1

        return
    
def Simulator():
    #CarVideo = cv2.VideoCapture('ew.mp4') # Cambiar por el dispositivo de cámara
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame,(900,600))
        controlkey = cv2.waitKey(1)
        if ret: #En esta seccion se detecta el carro       
            cars_frame = detect_cars(frame)
            
            cv2.imshow('frame', cars_frame)
            
            try:
                placa()
                print ("aqui toy")
            except Exception:
                print("Error")
        else:
            break
        if controlkey == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    Simulator()

