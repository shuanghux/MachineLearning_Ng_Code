function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%
% K = 3; X = 300*2 ; idx = 1*100; d = 2(dimensions)
% Useful variables
[m d] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, d); %3*2


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
count = zeros(1,K);
for i = 1:m
    for n = 1:K
        if idx(i) == n
           centroids(n,:) = centroids(n,:) + X(i,:);
           count(n) = count(n) + 1;
        end
    end
end
centroids = centroids./count';





% =============================================================


end

