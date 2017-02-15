clear;clc;
I = imread('1.jpg');
I_gray = rgb2gray(I);
I_resized = imresize(I_gray, 0.5);
I_noiseRed = imgaussfilt(I_resized);
I_refined = imadjust(I_noiseRed, stretchlim(I_noiseRed),[]);
imshow(I_refined);
imshow(medfilt(I_refined));
