% Auto Cutter for deep-apple-learning 
%
% Reads screenshots from 'input'-folder, extracts single apple images and
% stores them in 'output\rgb' and 'output\ir'
%
% First pre-processing step: 
% 1. Cut grid out of screenshot
% 2. Detect apples in grid
% 3. Build rectangle around apples and cut them out
% 4. Resize apple image and store it
%

%for each screenshot:
for i = 1:17

    %filenames:
    name_ir = ['input\', num2str(i),'i.png'];
    name_rgb = ['input\', num2str(i),'c.png'];
    %load images
    Iir = imread(name_ir);
    Irgb = imread(name_rgb);

    %Check certain region in image to find out the width of the grid:
    %If flag = true, then the grid is 7 apples wide, else 8.
    flag = (Iir(150:160,150:160) == 240);       

    %Crop grid:
    if(flag)
        Icrop = imcrop(Iir,[236 97 2403-236 1332-97]);
        Icrop_rgb = imcrop(Irgb,[236 97 2403-236 1332-97]);
    else
        Icrop = imcrop(Iir,[87 100 2552-87 1329-100]);
        Icrop_rgb = imcrop(Irgb,[87 100 2552-87 1329-100]);
    end

    %imshow(Icrop)

    %detect apples in grid:
    B = rgb2gray(Icrop) > 40;
    %remove red box in lower right corner:
    if(flag)
        B(1186:end,2108:end) = 0;
    else
       B(1183:end,2257:end) = 0;
       B(1122:1156,2454:end) = 0;
    end
    %erase grid lines:
    obj = strel('disk',5);
    B = imerode(imclose(B,obj),obj);
    %erase numbers on the left side:
    B = bwareaopen(B, 200);

    %imshow(B)

    %find bounding box of apples:
    info = regionprops(B,'Boundingbox');

    %extra pixels for each side of box:
    extra = 7;
    %%{
    %hold on
    for k = 1 : length(info)
         BB = info(k).BoundingBox;
         %rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2) ;

         %Get apple out of ir image:
         out_ir = imcrop(Icrop,[BB(1)-extra, BB(2)-extra, BB(3)+extra, BB(4)+extra]);
         out_ir = imresize(out_ir, [120,120]);
         %scale color intensity:
         out = double(out_ir);
         mx = max(out,[],'all');
         out_ir = uint8(out/mx*255);
         %sore image:
         name = ['output\ir\apple', num2str(i), '_', num2str(k),'.jpg'];
         imwrite(out_ir,name)

         %Get apple out of rgb image:
         out_rgb = imcrop(Icrop_rgb,[BB(1)-extra, BB(2)-extra, BB(3)+extra, BB(4)+extra]);
         out_rgb = imresize(out_rgb, [120,120]);
         %scale color intensity:
         out = double(out_rgb);
         mx = max(out,[],'all');
         out_rgb = uint8(out/mx*255);
         %store image:
         name = ['output\rgb\apple', num2str(i), '_', num2str(k),'.jpg'];
         imwrite(out_rgb,name)

    end
    %hold off
    %}
end
