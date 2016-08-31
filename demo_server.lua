local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.objdetect'
require 'cv.videoio'
require 'cv.imgproc'

local imname = {'./test1.jpg', './test2.jpg', './test3.jpg'}
local location = {{73,150,120,149},{40,271,120,145},{53,401,120,137}}

-- Load the cascades
local faceDetector = cv.CascadeClassifier{'./haarcascade_frontalface_alt.xml'}

local back = cv.imread {'./background.jpg', cv.IMREAD_COLOR}
-- modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
-- static const int componentsCount = 5
local bgdModel = torch.Tensor(1, 13*5):zero() 
local fgdModel = torch.Tensor(1, 13*5):zero()
local top, left, w, h = 26, 270, 160, 230

for num = 1, 3 do
    local frame = cv.imread {imname[num], cv.IMREAD_COLOR}
    local mask = torch.ByteTensor(frame:size(1), frame:size(2)):zero()
    local imgray = torch.ByteTensor(frame:size(1), frame:size(2))

    cv.cvtColor{frame, imgray, cv.COLOR_BGR2GRAY}
    mask:zero()

    -- Face detection   
    local faces = faceDetector:detectMultiScale{imgray}

    if (faces.size > 0) then
        local f = faces.data[1]
        local offset = 0
        mask[{ {f.y-offset, f.y+f.height+offset},{f.x-offset, f.x+f.width+offset} }] = 3
        cv.grabCut{img=frame, mask=mask, rect=0, bgdModel=bgdModel, fgdModel=fgdModel, iterCount=10, mode=cv.GC_INIT_WITH_MASK}
        local im = frame[{ {f.y-offset, f.y+f.height+offset},{f.x-offset, f.x+f.width+offset},{} }]
        local masktmp = mask[{ {f.y-offset, f.y+f.height+offset},{f.x-offset, f.x+f.width+offset} }]
        local top,left,w,h = location[num][1],location[num][2],location[num][3],location[num][4]
        
        local row, col = {top, top+h-1}, {left, left+w-1} 
        im = cv.resize{src = im, dsize = {w, h}}
        masktmp = cv.resize{src = masktmp, dsize = {w, h}}
        masktmp = masktmp:eq(3)

        for i = 1, 3 do
            local tmpim = im[{ {},{},i }]
            local tmpback = back[{ row,col,i}]
            tmpback[masktmp] = tmpim[masktmp]
            back[{ row,col,i}] = tmpback
        end
    end
end

cv.imwrite{'./process.jpg', back}




























