local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.objdetect'
require 'cv.imgproc'

function GrabCut(im)
    -- 输入为单张图像(RGB三通道), 输出为分割后的前景图像(RGBA四通道)
    -- 输出图像背景为透明颜色
    
    -- Load image
    if im:nDimension() == 0 then
        print('Problem loading image\n')
        os.exit(0)
    end
    
    local imgray = torch.ByteTensor(im:size(1), im:size(2))
    local imbgra = torch.ByteTensor(im:size(1), im:size(2), 4)
    cv.cvtColor{im, imgray, cv.COLOR_BGR2GRAY}
    cv.cvtColor{im, imbgra, cv.COLOR_BGR2BGRA}
    
    -- Load face detection xml
    local faceDetector = cv.CascadeClassifier{'data/haarcascade_frontalface_alt.xml'}
    
    -- Face detection
    local faces = faceDetector:detectMultiScale{imgray}
    
    local mask = torch.ByteTensor(im:size(1), im:size(2)):zero()
    local bgdModel = torch.Tensor(1, 13*5):zero() 
    local fgdModel = torch.Tensor(1, 13*5):zero()
    local f = faces.data[1] -- Only process one face image
    local offset = 0
    mask[{ {f.y-offset, f.y+f.height+offset},{f.x-offset, f.x+f.width+offset} }] = 3
    cv.grabCut{img=im, mask=mask, rect=0, bgdModel=bgdModel, fgdModel=fgdModel, iterCount=10, mode=cv.GC_INIT_WITH_MASK}
    mask = mask:eq(3)
    mask = mask:eq(0)

    for i = 1, 4 do
	    local tmp = imbgra[{ {},{},i }]
	    tmp[mask] = 0
	    imbgra[{ {},{},i }] = tmp
    end
    
    return imbgra
end

