% Image Processing Assignment
% George Monk, 17666456 ---------------------------------------------------

% Clearing variables and figures (formatting)
close all
clear 
clc

% Task 5: Robust method --------------------------

orig_img = imread('IMG_10.jpg'); % Orginal image

img_edit_name = 'IMG_10_Edit.jpg'; % Edited image name
ProcessImage(orig_img, img_edit_name, 1); % Function to convert image

% Task 6: Performance evaluation -----------------
% Step 1: Load ground truth data
GT = imread("IMG_10_GT.png"); % Reads GT image

L_GT = label2rgb(GT, 'flag', 'k'); % Visualises GT image using 'flag' colour scheme. Background set to black
L_GT = im2double(L_GT); % Converts GT to double

img = imread(img_edit_name); % Reads in edited image
img = imresize(img,(size(GT))); % Resizes it to ground truth image
img = im2double(img); % Converts it to double

figure, imshowpair(L_GT, img, 'Montage') % Shows images overlaid

ssimval = ssim(img, L_GT); % Calculates similarity index of both images
peaksnr = psnr(img, L_GT); % Calculates peak signal/noise ratio of both images

dice = 2*nnz(img&L_GT)/(nnz(img) + nnz(L_GT)); % Calculates dice score
% Calculates true/false positives/negatives
match_true = (img == L_GT) & img; 
match_false = (img == L_GT) & ~img;
match_true_x = img & ~L_GT;
match_false_x = ~img & L_GT;

headers = {'Positive', 'Negative'}; % Headers for table
% Total of true/false positives/negatives
truePos = sum(match_true(:));
trueNeg = sum(match_false(:));
falsePos = sum(match_true_x(:));
falseNeg = sum(match_false_x(:));

all = array2table([truePos trueNeg; falsePos, falseNeg]); % Converts to table for readability
all.Properties.VariableNames = headers;

precision = truePos/(truePos + falsePos); % Calculates precision
% Precision is calculated as:
% Number of true positives / (number of true positives + number of false
% positives)
recall = truePos/(truePos + falseNeg); % Calculates recall
% Recall is calculated as:
% Number of true positives / (number of true positives + number of false
% negatives)
trueNegative = trueNeg/(trueNeg + falseNeg); % True negative score.


accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg); % Calculates overall accuracy using precision & recall
fMeas = 2 * (precision * recall) / (precision + recall); % Calculates f-measure
function im = ProcessImage(input, name, save) % Converts image
    % Step-2: Covert image to grayscale
    im = rgb2gray(input); % Converts the image to greyscale.

    % Step-3: Rescale image
    I_bilinear = imresize(im,0.5,'bilinear'); % Rescales and binarises the image.
    % Step-4: Produce histogram before enhancing

    %figure, histogram(I_bilinear) % Produces the first histogram.   
    %title('Histogram of image before enhancement')
    
    %figure, imshow(I_bilinear)
    %title('Bilinear image before enhancement')
    I_en = imsharpen(I_bilinear, 'Amount', 5, 'Radius', 1); % The image is first sharpened to improve edge contrast. 
    %figure, imshow(I_en)
    %title('Image after first sharpen filter (2)')
    I_en = imflatfield(I_en, 35); % Flat field to normalise background
    %figure, imshow(I_en)
    %title('Image after flat field correction (3)')
    I_en = imgaussfilt(I_en); % Gaussian filter
    %figure, imshow(I_en)
    %title('Image after first gauss filter (4)')
    I_en = locallapfilt(I_en, 0.5, 0.8); % Local Laplacian filter enhances edges
    %figure, imshow(I_en)
    %title('Image after local Laplacian filtering (5)')
    I_rev = imcomplement(I_en); % To detect any foreground items that blend into the background, the reverse of i is generated.
    I_rev = imlocalbrighten(I_rev); % This is brightened,
    I_rev = imsharpen(I_rev, 'Amount', 5, 'Radius', 1); % sharpened,
    I_rev = imcomplement(I_rev); % and reversed again
    %figure, imshow(I_rev)
    %title('Incomplement of image after sharpen filter (6)')
    I_en = (I_en + I_rev) / 2; % Average of reverse and normal image
    %figure, imshow(I_en)
    %title('Image after incomplement adjustment and averaging (7)')
    I_en = imgaussfilt(I_en); % Guassian filter
    %figure, imshow(I_en)
    %title('Image after second gauss filter (8)')
    filtered = imbothat(I_en,(strel('disk', 15))); % Bottom hat filtering
    I_en = imadjust(filtered); % Adjusts image via bottom hat
    %figure, imshow(I_en)
    %title('Image after bottom hattting (9)')
    I_en = imsharpen(I_en, 'Amount', 0.8, 'Radius', 2); % Sharpens again
    %figure, imshow(I_en)
    %title('Image after final sharpen filter (10)')
    I_en = imgaussfilt(I_en); % Gaussian filter
    %figure, imshow(I_en)
    %title('Image after final gauss filter')
    % Step-6: Histogram after enhancement


    %figure, histogram(I_en)
    %title('Histogram of image after enhancement')
    % Step-7: Image Binarisation
    T = graythresh(I_en); % Otsu's threshold for binarisation
    b = imbinarize(I_en, T); % Binarises image
    b = bwareaopen(b, 16); % Removes pixels with at most 16 connected elements
    %figure, imshow(b)
    %title('Final binarised image after bwareopen and Otsu thresholding')
    % Task 2: Edge detection ------------------------
    b_edge = edge(b,'canny'); % Canny edge detection

    %figure, imshow(b_edge)
    %title('Binary image after canny edge detection (13)')
    % Task 3: Simple segmentation --------------------
    %se90 = strel('line',4,90);  % Alternative segmentation method
    %se0 = strel('line',4,0);
    %BWsdil = imdilate(BWs,[se90 se0]);
    %BWdfill = imfill(BWsdil,'holes');

    mrph = bwmorph(b_edge, 'close'); % Closes edges
    mrphfill = imfill(mrph,4,'holes'); % Fills shapes
    figure, imshow(mrphfill)
    title('Binary image after image filling')
    %mrphfill = bwmorph(mrphfill, 'thick', 1);

    % Task 4: Object Recognition --------------------

    %bw2 = ~bwareaopen(~BWdfill, 10);
   
    
    disk = strel('disk', 2); % Small erosion filter
    test = imerode(mrphfill, disk);
    skel = bwmorph(test, 'skeleton', Inf); % Creates skeleton
    skel = bwmorph(skel, 'spur', 20); % 'Prunes' the skeleton by reducing length of all skeletons by 20
    brk = bwmorph(skel, 'branchpoints'); % Finds branchpoints
    
    %skel = skel - brk; % Removes from skeleton
    skel = bwmorph(skel, 'diag'); % Closes skeleton diagonally and horizontally
    skel = bwmorph(skel, 'bridge');
    skel = bwmorph(skel, 'thicken', 24); % Thickens skeleton
    skel = bwmorph(skel, 'open'); % Opens curvepoints
    skel = bwmorph(skel, 'thin', 9); % Thins to accentuate curves
    brk = bwmorph(brk, 'thicken', 1); % Thickens branchpoints
    skel = skel - brk; % Removes branchpoints from skeleton

    D = -bwdist(~skel); % Distance transform of skeleton
    mask = imextendedmin(D,9); % Extended minima transformation
    D2 = imimposemin(D,mask); % Applies to distance transform
    Ld2 = watershed(D2); % Watershed transformation to segment regions of high intensity
    im_segmented = mrphfill; % Image to segment
    im_segmented(Ld2 == 0) = 0; % Where watershed has been applied to skeleton, apply to original image
    
    im_segmented = bwareaopen(im_segmented, 120); % Remove objects that may have been sliced by watershed
    %figure, imshowpair(im_segmented, skel, 'Montage')   
    %title('Comparison between segmented image and its skeleton')
    
    [B,L] = bwboundaries(im_segmented,'noholes'); % Traces region boundaries
    stats = regionprops(L,'Area','Centroid'); % Uses boundaries to compute the area and centroid of each object

    figure (1), imshow(L) % Final image
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