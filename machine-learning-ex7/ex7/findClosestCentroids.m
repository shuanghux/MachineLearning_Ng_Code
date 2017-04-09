function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K as number of centroids
K = size(centroids, 1);
% Set m as number of samples
m = size(X,1);
% Set d as sample dimentions
d = size(X,2);
% You need to return the following variables correctly.


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

X = X'; % 2*300 d*m
centroids = centroids';%2*3 d*K
distances = zeros(K,m);%3*300 K*m

for i = 1:K
   vect = X - centroids(:,i); %2*300
   vect = sum(vect.^2); %1*300
   distances(i,:) = vect;
end
[val,idx] = min(distances);




% =============================================================

end

