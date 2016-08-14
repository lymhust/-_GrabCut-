local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.highgui'
require 'cv.objdetect'
require 'cv.imgproc'
require 'image'

-- Load image
local im = cv.imread {'data/test1.jpg', cv.IMREAD_COLOR}
if im:nDimension() == 0 then
    print('Problem loading image\n')
    os.exit(0)
end

-- Resize
--[[
local imsize = 128
local imsmall = torch.ByteTensor(imsize, imsize, 3)
cv.resize{src=im, dsize=cv.Size{imsize, imsize}, dst=imsmall}
im = imsmall
--]]

local imgray = torch.ByteTensor(im:size(1), im:size(2))
local imbgra = torch.ByteTensor(im:size(1), im:size(2), 4)
cv.cvtColor{im, imgray, cv.COLOR_BGR2GRAY}
cv.cvtColor{im, imbgra, cv.COLOR_BGR2BGRA}
--imbgra[{ {},{},4 }] = 128
--cv.equalizeHist(imgray, imgray)

-- Load the cascades and detect face
local faceDetector = cv.CascadeClassifier{'data/haarcascade_frontalface_alt.xml'}
local faces = faceDetector:detectMultiScale{imgray}
						
local imcopy = im:clone()												
for i = 1, faces.size do
   local f = faces.data[i]
   cv.rectangle{imcopy, {f.x, f.y}, {f.x+f.width, f.y+f.height}, color={0,255,0}, thickness=5}
end
cv.imshow {"Input image", imcopy}
cv.imwrite{'data/face_detected.jpg', imcopy}
--cv.waitKey {0}

-- GrabCut
--cv.GC_BGD = 0
--cv.GC_EVAL = 2
--cv.GC_FGD = 1
--cv.GC_INIT_WITH_MASK = 1
--cv.GC_INIT_WITH_RECT = 0
--cv.GC_PR_BGD = 2
--cv.GC_PR_FGD = 3
local mask = torch.ByteTensor(im:size(1), im:size(2)):zero()
-- modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
-- static const int componentsCount = 5
local bgdModel = torch.Tensor(1, 13*5):zero() 
local fgdModel = torch.Tensor(1, 13*5):zero()
local f = faces.data[1]
local offset = 0
mask[{ {f.y-offset, f.y+f.height+offset},{f.x-offset, f.x+f.width+offset} }] = 3
cv.grabCut{img=im, mask=mask, rect=0, bgdModel=bgdModel, fgdModel=fgdModel, iterCount=10, mode=cv.GC_INIT_WITH_MASK}

mask = mask:eq(3)
mask = mask:eq(0)
--image.display(mask)

for i = 1, 4 do
	local tmp = imbgra[{ {},{},i }]
	tmp[mask] = 0
	imbgra[{ {},{},i }] = tmp
end
cv.imwrite{'data/face_segmented.png', imbgra}
-- Show image
cv.imshow{"Mask", imbgra}
cv.waitKey {0}
print('Finished')


























