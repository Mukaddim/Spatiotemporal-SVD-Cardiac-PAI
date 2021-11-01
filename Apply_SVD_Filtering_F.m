function [Output_IQ,U,D,V,noise_rank] = Apply_SVD_Filtering_F(Input_IQ,static_rank,grad_thr)
    
    % Inputs: Input_IQ = strcuture holding 3-D (2D space and 1D time) DAS data in IQ format
    % static_rank = r_st in the JBO paper (used for lower order SVD thresholding)
    % grad_thr = threshold for gradient turning point (high cut-off)
  
    
    %% ----- Global SpatioTemporal SVD ------- %%
    [samples,lines,totalTime] = size(Input_IQ.DAS);

    % ---- Casorati Matrix Fomration ---- %
    casorati_mat = complex(zeros(samples*lines,totalTime));
    for k = 1:totalTime
        imgdata = Input_IQ.DAS(:,:,k);
        casorati_mat(:,k) = imgdata(:);
    end

    % --- SVD ---%
    tic;
    [U,D,V]= svd(casorati_mat,0);
    toc;
    
    % --- Estimate Gradient --- %
    sig_values = diag(D);
    gradient_sig = gradient(sig_values);
    noise_rank = find(abs(gradient_sig)<grad_thr,1,'first');

    sig_values_F = sig_values;
    sig_values_F(1:static_rank) = 0;
    sig_values_F(noise_rank:end) = 0;
    D_Filtered = diag(sig_values_F);

    casorati_mat_F = U*D_Filtered*V';

    for k=1:totalTime
        filterimgdata = casorati_mat_F(:,k);
        filterimgdata = reshape(filterimgdata,[samples,lines]);
        Output_IQ.DAS(:,:,k) = filterimgdata;
        

    end

end

