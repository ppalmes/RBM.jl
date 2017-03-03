module RBM
using Winston

export dream
export showWeights

function xp(x)
    return 1.0 ./ (1+exp(-x))
end

function vtohprob{T<:Real,P<:Integer}(rbmw::Matrix{T},vstate::Matrix{P})
    return xp(rbmw * vstate)
end

function htovprob{T<:Real,P<:Integer}(rbmw::Matrix{T},hstate::Matrix{P})
    return xp(rbmw' * hstate)
end

function gradient{T<:Real,P<:Integer}(vstate::Matrix{P},hstate::Matrix{T})
    hstate * vstate'/size(vstate)[2]
end

function bernoulli{T<:Real}(v::Vector{T})
    return map((x)->x>rand()?1:0,v)
end

function bernoulli{T<:Real}(m::Matrix{T})
    return map((x)->x>rand()?1:0,m)
end

function cd{T<:Real}(rbmw::Matrix{T},data::Matrix{T})
    vdata=bernoulli(data)
    h0=bernoulli(vtohprob(rbmw,vdata))
    vh0=gradient(vdata,h0)
    v1=bernoulli(htovprob(rbmw,h0))
    h1=bernoulli(vtohprob(rbmw,v1))
    vh1=gradient(v1,h1)
    return vh0-vh1
end

function showWeights{T<:Real}(W::Matrix{T},row,col)
    r,c = size(W)
    plots = []
    for i in 1:r
        plots=vcat(plots,Winston.imagesc((reshape(W[i,:],28,28))'))
    end
    
    mgrid=Winston.Table(row,col)
    c=1
    for i = 1:row
        for j=1:col
            mgrid[i,j] = plots[c]
            c+=1
        end
    end  
    display(mgrid)
end

function dream{T<:Real}(rbmw::Matrix{T},data::Matrix{T})
    vdata=bernoulli(data)
    h0=bernoulli(vtohprob(rbmw,vdata))
    v1=bernoulli(htovprob(rbmw,h0))
  
    rw = size(vdata)[2]
    mgrid = Winston.Table(1,2)
    for i in 1:rw
        p1=Winston.imagesc(reshape(vdata[:,i],28,28)')
        p2=Winston.imagesc(reshape(v1[:,i],28,28)')
        mgrid[1,1]=p1
        mgrid[1,2]=p2
        display(mgrid)
    end
    return v1
end

function rbm{T<:Real,P<:Integer}(nhid::P,data::Matrix{T},lr::T,niter::P,mbatchsz::P,mom::T)
    psz,nsz=size(data)
    momentum=mom
    if nsz % mbatchsz != 0
        println("test cases must be multiple of batch size")
        return
    end
    model=(rand(nhid,psz)*2-1)*0.1
    momSpeed=zeros(nhid,psz)

    startMini=1
    for iter in 1:niter
        if mod(iter,50)==0
            println("iter:",iter," ",sqrt(mean(map((x)->x*x,model))))
        end
        mbatch=data[:,startMini:(startMini+mbatchsz-1)]
        startMini=(startMini+mbatchsz) % nsz
        gradient=cd(model,mbatch)
        momSpeed=momentum*momSpeed+gradient
        model=model+momSpeed*lr
    end
    return model
end
end

using RBM
#trainData=readdlm("./mnist_train.csv",',',Float64)
#trainLab=trainData[:,1]
#trainData=trainData[:,2:size(trainData)[2]]/255.0
#trainData=trainData'

testData=readdlm("./mnist_test.csv",',',Float64)
testLab=testData[:,1]
testData=testData[:,2:size(testData)[2]]/255.0
testData=testData'

hidSz=30
mbatchsz=100
#niter=20000
niter = 500
lr=0.01
mom=0.9

#W=RBM.rbm(hidSz,trainData,lr,niter,mbatchsz,mom)

W=readcsv("./weighnobias.csv")

RBM.dream(W,testData[:,1:50]);
