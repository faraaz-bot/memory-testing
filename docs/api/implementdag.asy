
import flowchart;
//defaultpen(fontsize(8pt));
size(14cm,0);

real dx=2;
real dy=1;

// from left to right and top to bottom
block spectralapi = roundrectangle(Label("Spectral Ops API"), (-dx,0));
block multiapi    = roundrectangle(Label("MultiGPU API"),     (0,0));
block sparseapi   = roundrectangle(Label("Sparsity API"),     (dx,0));

block api         = roundrectangle(Label("API Proposal"),     (0,-dy));

block oneway = roundrectangle(Label("One-way ops"), (-dx,-2dy));
block twoway = roundrectangle(Label("Round-trip unpadded ops"), (-dx,-3dy));
block twowaypad = roundrectangle(Label("Round-trip padded ops"), (-dx,-4dy));

block blockdata = roundrectangle(Label("Block decomposition"), (0,-2dy));

block singlenode = roundrectangle(Label("Single-node multi-gpu"), (0,-3dy));
block multinode = roundrectangle(Label("Multi-node MPI"), (0,-4dy));
block multinodeschmem = roundrectangle(Label("Multi-node SCHMEM"), (0,-5dy));

block sparse = roundrectangle(Label("Sparse data"), (dx,-3dy));

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
    
    spectralapi--Down--Arrow--Right--api;
    multiapi--Down--Arrow--Down--api;
    sparseapi--Down--Arrow--Left--api;

    draw(pic, api.bottom(t)--oneway.top(t), Arrow);
    draw(pic, oneway.bottom(t)--twoway.top(t), Arrow);
    draw(pic, twoway.bottom(t)--twowaypad.top(t), Arrow);
    
    //draw(pic, api.bottom(t)--oneway.top(t), Arrow);
    draw(pic, api.bottom(t)--blockdata.top(t), Arrow);

    draw(pic, blockdata.bottom(t)--singlenode.top(t), Arrow);
    draw(pic, singlenode.bottom(t)--multinode.top(t), Arrow);
    draw(pic, multinode.bottom(t)--multinodeschmem.top(t), Arrow);
    
    draw(pic, blockdata.bottom(t)--sparse.top(t), Arrow);
  });
