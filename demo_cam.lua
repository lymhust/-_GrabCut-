local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.objdetect'
require 'cv.videoio'
require 'cv.imgproc'


local cap = cv.VideoCapture{device=0}
if not cap:isOpened() then
    print("Failed to open the default camera")
    os.exit(-1)
end

cv.namedWindow{"frame", cv.WINDOW_AUTOSIZE}
cv.namedWindow{"processed", cv.WINDOW_AUTOSIZE}
local _, frame = cap:read{}

local imgray = torch.ByteTensor(frame:size(1), frame:size(2))

-- Load the cascades
local faceDetector = cv.CascadeClassifier{'data/haarcascade_frontalface_alt.xml'}

local background = cv.imread {'data/background.jpg', cv.IMREAD_COLOR}
local back = background:clone()
local mask = torch.ByteTensor(frame:size(1), frame:size(2)):zero()
-- modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
-- static const int componentsCount = 5
local bgdModel = torch.Tensor(1, 13*5):zero() 
local fgdModel = torch.Tensor(1, 13*5):zero()
local location = {{73,150,120,149},{40,271,120,145},{53,401,120,137}}
local num = 1

while true do
    
    cv.cvtColor{frame, imgray, cv.COLOR_BGR2GRAY}
    mask:zero()
    
    -- Face detection   
    local faces = faceDetector:detectMultiScale{imgray}
    local imcopy = frame:clone()												
    for i = 1, faces.size do
       local f = faces.data[i]
       cv.rectangle{imcopy, {f.x, f.y}, {f.x+f.width, f.y+f.height}, color={0,255,0}, thickness=5}
    end
    
    if cv.waitKey{30} >= 0 then
        if (faces.size > 0) then
        
            if (num>3) then num = 1 back = background:clone() end
            print(faces.size)
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
            num = num + 1
        end
        cv.imshow{"processed", back}
    end
    
    cv.imshow{"frame", imcopy}
    

    --if cv.waitKey{30} >= 0 then break end

    cap:read{frame}
end





























