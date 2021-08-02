% Image Processing Assignment
% George Monk, 17666456 ---------------------------------------------------
% Clearing variables and figures (formatting)
clear; 
close all;
clc
% Task 1: Pre-processing -----------------------
% Step-1: Load input image
I = imread('IMG_01.jpg'); % Reads the designated image.

ProcessImage(I, 'IMG_01_Edit.jpg', 0); % Function to convert image 

function im = ProcessImage(input, name, save) % Converts image
    % Step-2: Covert image to grayscale
    im = rgb2gray(input); % Converts the image to greyscale.

    % Step-3: Rescale image
    I_bilinear = imresize(im,0.5,'bilinear'); % Rescales and binarises the image.
    % Step-4: Produce histogram before enhancing
    figure, imshow(I_bilinear)
    title('Resized Image')
    figure, histogram(I_bilinear) % Produces the first histogram.   
    title('Histogram of image before enhancement')
    image = adapthisteq(I_bilinear); % Adaptive histogram enhancement
    image = imsharpen(image, 'Amount', 5, 'Radius', 1); % Sharpens image
    image = imgaussfilt(image); % Applies gaussian filter
    image_rev = imcomplement(image); % To detect any foreground items that blend into the background, the inverse of i is generated.
    image_rev = imsharpen(image_rev); % Sharpens inverse
    image_rev = imcomplement(image_rev); % Inverse again
    image = (image + image_rev) / 2; % Generates average from inverse and original
        
    image = imflatfield(image, 20); % Flat field 
 
    image = localcontrast(image); % Local contrast enhancement
    image = imgaussfilt(image); % Gaussian filter
    image = imsharpen(image, 'Amount', 6); % Sharpens image
    
    figure, imshow(image)
    title('Enhanced Image')
    
    % Step-6: Histogram after enhancement
    figure, histogram(image)
    title('Histogram of image after enhancement')
    % Step-7: Image Binarisation
    T = graythresh(image); % Otsu's method to threshold for binarisation
    b = ~imbinarize(image, T); % Binarises image
    b = bwareaopen(b, 9); % Eliminates pixels with no more than 9 connected elements
    figure, imshow(b)
    title('Final binarised image after bwareopen and Otsu thresholding')
    % Task 2: Edge detection ------------------------
    b_edge = edge(b,'canny'); % Canny edge detection
 
    figure, imshow(b_edge)
    title('Binary image after canny edge detection')
    % Task 3: Simple segmentation --------------------
    se90 = strel('line',4,90); % Produces strels to dilate pixels by four
    se0 = strel('line',4,0);
    BWsdil = imdilate(b_edge,[se90 se0]); % Dilates image
    BWdfill = imfill(BWsdil,'holes'); % Fills holes
    
    figure, imshow(BWdfill)
    title('Binary image after image filling')
    %mrphfill = bwmorph(mrphfill, 'thick', 1);

    % Task 4: Object Recognition --------------------

    D = -bwdist(~BWdfill); % Distance transform of image
    mask = imextendedmin(D,6.5); % Extended minima transformation
    D2 = imimposemin(D,mask); % Applies to distance transform
    Ld2 = watershed(D2); % Watershed transformation to segment regions of high intensity
    im_segmented = BWdfill; % Image to segment
    im_segmented(Ld2 == 0) = 0; % Applies watershed transformation to iamge
    
    figure, imshow(im_segmented)   
    title('Segmented Image')
    
    [B,L] = bwboundaries(im_segmented,'noholes');
    stats = regionprops(L,'Area','Centroid');

    figure (1), imshow(L)
    %title('Results of prior stages and object recognition') % Adding title
    %harms final results as it is added to output image
    % For every boundary (object) in image
    for k = 1:length(B)

        % Get boundary coordinates
        boundary = B{k};

        % Estimate perimeter
        delta_sq = diff(boundary).^2;    
        perimeter = sum(sqrt(sum(delta_sq,2)));

        % Using regionprops, get area
        area = stats(k).Area;

        % Calculate roundness of object 'B{k}' using it's area and
        % perimeter
        metric = 4*pi*area/perimeter^2;

        % Display results
        hold on
        if metric >= 0.35 && metric < 0.83 % Short screw
            % Fill boundary region with appropriate colour to designate
            % shape.
            fill(boundary(:,2), boundary(:,1), 'white', 'LineWidth', 2);
        else % Long screw
            fill(boundary(:,2), boundary(:,1), 'blue', 'LineWidth', 2)
        end
        if metric >= 0.83 % Washer
            fill(boundary(:,2), boundary(:,1), 'red', 'LineWidth', 2)
        end  
    end
    if save == 1 % If image has been set to save
        % Export the figure with 'fill' additions
        exportgraphics(figure (1), name, 'Resolution', 150)
    end
end
