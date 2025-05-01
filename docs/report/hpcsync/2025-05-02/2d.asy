size(100,100);


int N = 16;
int P = 4;

real w = 0.1;
real d = 0.1;

for(int ipic = 0; ipic < 4; ++ipic)
{
    picture pic;
    size(pic,175,0);

    if(ipic == 0) {
        for(int px = 0; px < P; ++px) {
            pair b0 = (px * w + d * (N#P) * px , 0);
            pair b1 = (px * w + d * (N#P) * (px + 1) , 0);
            pair b2 = (px * w + d * (N#P) * (px + 1) , d * N);
            pair b3 = (px * w + d * (N#P) * (px) , d * N);

            draw(pic, b0--b1--b2--b3--cycle, Pen(px));
    
            for(int ix = 0; ix < N # P; ++ix) {
                for(int iy = 0; iy < N; ++iy) {
                    pair q = (d* ((ix + 0.5) + (N#P) * px) + w * px,
                              d*(iy + 0.5));
                    dot(pic, q);
                }
            }          
        }
    }

    if(ipic == 1) {
        for(int px = 0; px < P; ++px) {
            for(int py = 0; py < P; ++py) {
                pair b0 = (px * w + d * (N#P) * px ,
                           py * w + d * (N#P) * py);
                pair b1 = b0 + (N#P * d , 0);
                pair b2 = b0 + (N#P * d , N#P * d );
                pair b3 = b0 + (0 , N#P * d );
                      
                draw(pic, b0--b1--b2--b3--cycle, Pen(px));
    
                for(int ix = 0; ix < N # P; ++ix) {
                    for(int iy = 0; iy < N # P; ++iy) {
                        pair q = b0 + ((ix + 0.5) * d, (iy+ 0.5) * d);
                        dot(pic, q);
                    }
                }          
            }
        }
    }

    if(ipic == 2) {
        for(int px = 0; px < P; ++px) {
            for(int py = 0; py < P; ++py) {
                pair b0 = (px * w + d * (N#P) * px ,
                           py * w + d * (N#P) * py);
                pair b1 = b0 + (N#P * d , 0);
                pair b2 = b0 + (N#P * d , N#P * d );
                pair b3 = b0 + (0 , N#P * d );
                      
                draw(pic, b0--b1--b2--b3--cycle, Pen(py));
    
                for(int ix = 0; ix < N # P; ++ix) {
                    for(int iy = 0; iy < N # P; ++iy) {
                        pair q = b0 + ((ix + 0.5) * d, (iy+ 0.5) * d);
                        dot(pic, q);
                    }
                }          
            }
        }
    }

    if(ipic == 3) {
        for(int py = 0; py < P; ++py) {
            pair b0 = (0,
                       py * w + d * (N#P) * py);
            pair b1 = b0 + (N * d , 0);
            pair b2 = b0 + (N * d , N#P * d );
            pair b3 = b0 + (0 , N#P * d );
                      
            draw(pic, b0--b1--b2--b3--cycle, Pen(py));
            
            for(int ix = 0; ix < N; ++ix) {
                for(int iy = 0; iy < N # P; ++iy) {
                    pair q = b0 + ((ix + 0.5) * d, (iy+ 0.5) * d);
                    dot(pic, q);
                }
            }
        }
    }

    shipout("2d"+(string) ipic, pic);
}
