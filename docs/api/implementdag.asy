
import flowchart;
//defaultpen(fontsize(8pt));
size(18cm, 0);

real dx=2;
real dy=1;

real xop = -2.5dx;
real xmulti = 0;
real xsparse = 2dx;

// from left to right and top to bottom
block spectralapi = roundrectangle(Label("Spectral Ops API"), (xop,0));
block multiapi    = roundrectangle(Label("MultiGPU API"),     (0,0));
block sparseapi   = roundrectangle(Label("Sparsity API"),     (xsparse,0));

block api         = roundrectangle(Label("API Proposal"),     (0,-dy));

block oneway = roundrectangle(Label("One-way ops"), (xop,-2dy));
block twoway = roundrectangle(Label("Round-trip unpadded ops"), (xop,-3dy));
block twowaypad = roundrectangle(Label("Round-trip padded ops"), (xop,-4dy));

block blockdata = roundrectangle(Label("Block decomposition"), (0,-2dy));

block singlenode = roundrectangle(Label("Single-node multi-gpu"), (-0.8dx,-3dy));
block multinode = roundrectangle(Label("Multi-node MPI"), (0,-4dy));
block multinodeschmem = roundrectangle(Label("Multi-node SCHMEM"), (dx,-5dy));

block sparse = roundrectangle(Label("Sparse data"), (xsparse,-3dy));

// draw the blocks
draw(spectralapi);
draw(multiapi);
draw(sparseapi);

draw(api);

draw(oneway);
draw(twoway);
draw(twowaypad);

draw(blockdata);
draw(singlenode);
draw(multinode);
draw(multinodeschmem);

draw(sparse);


// draw connections (as blocks are drawn)
add(new void(picture pic, transform t) {
    blockconnector operator --=blockconnector(pic, t);
    
    draw(pic, spectralapi.bottomright(t)--api.topleft(t), Arrow);
    draw(pic, multiapi.bottom(t)--api.top(t), Arrow);
    draw(pic, sparseapi.bottomleft(t)--api.topright(t), Arrow);

    draw(pic, api.bottomleft(t)--oneway.topright(t), Arrow);
    draw(pic, oneway.bottom(t)--twoway.top(t), Arrow);
    draw(pic, twoway.bottom(t)--twowaypad.top(t), Arrow);
    
    draw(pic, api.bottom(t)--blockdata.top(t), Arrow);

    draw(pic, blockdata.position(-1.3, t)--singlenode.top(t), Arrow);
    draw(pic, blockdata.bottom(t)--multinode.top(t), Arrow);
    draw(pic, blockdata.position(-1.7, t)--multinodeschmem.top(t), Arrow);
    
    draw(pic, blockdata.bottomright(t)--sparse.top(t), Arrow);
  });
