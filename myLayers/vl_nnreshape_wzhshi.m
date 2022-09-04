function y = vl_nnreshape_wzhshi( x, dims, dzdy )

assert(sum(dims == -1) <= 1, 'at most one dim can be computed from the others') ;
assert(length(dims) == 3, 'dims should have three elements') ;

sz = size(x) ;

copyDims = find(dims == 0) ;
if copyDims
    dims(copyDims) = sz(copyDims) ;
end

targetDim = find(dims == -1) ;
if targetDim
    idx = [1 2 3] ;
    idx(targetDim) = [] ;
    dims(targetDim) = prod(sz(1:3)) / prod(dims(idx)) ;
end

dims = horzcat(dims, size(x,4)) ;

if nargin <= 2 || isempty(dzdy)
    y = reshape(x, dims) ;
else
    y = reshape(dzdy, size(x)) ;
end
