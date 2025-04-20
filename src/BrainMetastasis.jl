module BrainMetastasis

using DataFrames
using Dates
using CSV
using Statistics
using Plots, StatsPlots
using MLJ
using Logging
using StatisticalMeasuresBase
using Shapley, CategoricalDistributions
import Imbalance

Jaccard(cm) = truepositive(cm) / (truepositive(cm) + falsenegative(cm) + falsepositive(cm))
Jaccard(ŷ, y) = truepositive(ŷ, y) / (truepositive(ŷ, y) + falsenegative(ŷ, y) + falsepositive(ŷ, y))

struct JaccardIndex
end
(measure::JaccardIndex)(cm) = Jaccard(cm)
(measure::JaccardIndex)(ŷ, y) = Jaccard(ŷ, y)
@trait JaccardIndex orientation = Score()

value_counts(df, col) = combine(groupby(df, col), nrow)

function mkDate(s::Union{Missing, AbstractString})
    if ismissing(s) || match(r"\d\d.\d\d.\d\d\d\d", s) === nothing
        return missing
    end
    return Date(s , dateformat"dd.mm.yyyy")
end

function dataRename!(df::DataFrame)
    rename!(df, [:Sex,
                 :Birthday,
                 :Diagnosis,
                 :DiagnosisDate,
                 :OperationDate,
                 :ExpansionDate,
                 :Mutations,
                 :BrainRadiationDate,
                 :BrainOperationDate,
                 :Fractions,
                 :RadiotherapyDate,
                 :KarnovskyIndex,
                 :NidusNumber,
                 :NidusVolume,
                 :MaxNidusVolume,
                 :ExtraMetastasis,
                 :Medicament,
                 :Recidivism,
                 :DistantMetastasis,
                 :IntraProgression])
end

function dateEdit!(df::DataFrame)
    transform!(df, :Birthday => (s -> mkDate.(s)) => :Birthday)
    transform!(df, :DiagnosisDate => (s -> mkDate.(s)) => :DiagnosisDate)
    transform!(df, :ExpansionDate => (s -> mkDate.(s)) => :ExpansionDate)
    transform!(df, :RadiotherapyDate => (s -> mkDate.(s)) => :RadiotherapyDate)
end

function oneHotEncoding!(df::DataFrame)
    transform!(df, :Diagnosis => (s -> ifelse.(s .== "РМЖ", 1, 0)) => :RMG)
    transform!(df, :Diagnosis => (s -> ifelse.(s .== "РП", 1, 0)) => :RP)
    transform!(df, :Diagnosis => (s -> ifelse.(s .== "НМРЛ", 1, 0)) => :NMRL)
    transform!(df, :Diagnosis => (s -> ifelse.(s .== "Меланома", 1, 0)) => :Melanome)
    transform!(df, :Diagnosis => (s -> ifelse.(s .== "КРР", 1, 0)) => :KRR)
    
    if(size(df, 1) == sum(df.RMG) + sum(df.RP) + sum(df.NMRL) + sum(df.Melanome) + sum(df.KRR))
        println("OneHotEncoding Diagnosis OK")
    end
    select!(df, Not(:Diagnosis))

    transform!(df, :Medicament => (s -> ifelse.(s .=== missing, "Missing", s)) => :Medicament)
    transform!(df, :Medicament => (s -> ifelse.(s .== "Химиотерапия", 1, 0)) => :Chemotherapy)
    transform!(df, :Medicament => (s -> ifelse.(s .== "Таргетная терапия", 1, 0)) => :TargetTherapy)
    transform!(df, :Medicament => (s -> ifelse.(s .== "Без лечения", 1, 0)) => :WithoutMedicament)
    if(sum(s != "Missing" for s in df[!, :Medicament]) == 
        sum(df.Chemotherapy) + sum(df.TargetTherapy) + sum(df.WithoutMedicament))
        println("OneHotEncoding Medicament OK")
    end
    select!(df, Not(:Medicament))
end

function intraProgressionCheck!(R::Union{Missing, AbstractString}, D::Union{Missing, AbstractString})
    if ismissing(R) && ismissing(D) 
        return 0
    end
    if R != "нет" || D != "нет"
        return 1
    else
        return 0
    end
end

function sexEdit!(df::DataFrame)
    transform!(df, :Sex => (s -> uppercase.(s)) => :Sex)
    transform!(df, :Sex => (s -> ifelse.(s .== "М", 0, 1)) => :Sex)
end

function volumeEdit(s::AbstractString)
    s = replace(s, "," => ".")
    parse(Float64, s)
end

function correlationMatrix(df::DataFrame)
    M = cor(Matrix(df))
    cols = names(df)
    (n,m) = size(M)
    heatmap(M, fc=cgrad(:seismic), xticks=(1:m,cols), clim=(-1, 1),
            xrot=90, yticks=(1:m,cols), yflip=true, size=(1200, 1200))
    annotate!([(j, i, text(round(M[i,j],digits=3), 8,"Computer Modern",:black)) for i in 1:n for j in 1:m])
end

function h(X_i::Vector, X_j::Vector, q_2::Real)
    res = []
    for x_i in X_i
        for x_j in X_j
            if(x_i == x_j)
                push!(res, 0.0)
            else
                push!(res, (x_i + x_j - 2q_2)/(x_j - x_i))
            end
        end
    end
    return res
end

function adjNormalize(v::Vector)
    med = median(v)

    q_1 = quantile(v, 0.25)
    q_2 = quantile(v, 0.5)
    q_3 = quantile(v, 0.75)
    iqr = q_3 - q_1

    mc = median(h(v[v .<= q_2], v[v .>= q_2], q_2))

    l, r = 0, 0
    if mc >= 0
        l = q_1 - 1.5exp(-4mc)iqr
        r = q_3 + 1.5exp(3mc)iqr
    else
        l = q_1 - 1.5exp(-3mc)iqr
        r = q_3 + 1.5exp(4mc)iqr
    end

    existing_border_left = minimum(v[v .>= l])
    existing_border_right = maximum(v[v .<= r])

    if existing_border_left == existing_border_right
        existing_border_right = quantile(v, 0.95)
        existing_border_left = quantile(v, 0.05)
    end
    if existing_border_left == existing_border_right
        existing_border_right = maximum(v)
        existing_border_left = minimum(v)
    end
    adjusted_scale_value = existing_border_right - existing_border_left
    if adjusted_scale_value == 0.0
        adjusted_scale_value = 1
    end

    v = (v .- med)./adjusted_scale_value
end

function normalize!(df::DataFrame)
    transform!(df, :Fractions => (v -> adjNormalize(v)) => :Fractions)
    transform!(df, :KarnovskyIndex => (v -> adjNormalize(v)) => :KarnovskyIndex)
    transform!(df, :NidusNumber => (v -> adjNormalize(v)) => :NidusNumber)
    transform!(df, :NidusVolume => (v -> adjNormalize(v)) => :NidusVolume)
    transform!(df, :Age => (v -> adjNormalize(v)) => :Age)
    transform!(df, :ReactionTime => (v -> adjNormalize(v)) => :ReactionTime)
    transform!(df, :MetastasisTime => (v -> adjNormalize(v)) => :MetastasisTime)
end

function CSVPretrain(s::String)
    df = CSV.read(s, DataFrame)
    dataRename!(df)
    filter!(:IntraProgression => s -> !ismissing(s), df)
    transform!(df, [:Recidivism, :DistantMetastasis] => 
        ByRow((R, D) -> intraProgressionCheck!(R, D)) => :Progression)
    select!(df, Not([:Recidivism, :DistantMetastasis, :IntraProgression]))
    sexEdit!(df)
    select!(df, Not([:OperationDate, :Mutations, :ExtraMetastasis]))

    transform!(df, :BrainRadiationDate => 
        (s -> ifelse.(ismissing.(s) .|| s .== "нет", 0, 1)) => :BrainRadiation)
    select!(df, Not(:BrainRadiationDate))
    transform!(df, :BrainOperationDate => 
        (s -> ifelse.(ismissing.(s) .|| s .== "нет", 0, 1)) => :BrainOperation)
    select!(df, Not(:BrainOperationDate))
    
    oneHotEncoding!(df)
    transform!(df, :NidusVolume => (s -> volumeEdit.(s)) => :NidusVolume)
    transform!(df, :MaxNidusVolume => (s -> volumeEdit.(s)) => :MaxNidusVolume)

    filter!(:DiagnosisDate => s -> !ismissing(s), df)
    filter!(:ExpansionDate => s -> !ismissing(s), df)
    filter!(:ExpansionDate => s -> match(r"\d\d.\d\d.\d\d\d\d", s) !== nothing, df)
    dateEdit!(df)
    transform!(df,[:RadiotherapyDate, :Birthday] => ((R, B) -> Dates.value.(R .- B)) => :Age)
    transform!(df,[:RadiotherapyDate, :ExpansionDate] => ((R, E) -> Dates.value.(R .- E)) => :ReactionTime)
    transform!(df,[:ExpansionDate, :DiagnosisDate] => ((E, D) -> Dates.value.(E .- D)) => :MetastasisTime)
    select!(df, Not([:Birthday, :DiagnosisDate, :ExpansionDate, :RadiotherapyDate]))

    select!(df, Not(:MaxNidusVolume))
    select!(df, Not(:Progression), :Progression)

    normalize!(df)

    # savefig(correlationMatrix(df), "corrMatrix.png")

    CSV.write("pretrained.csv", df)
end

function DataPrepare(s::String; balancing="no")
    df = CSV.read(s, DataFrame)
    df = coerce(df, :Progression => OrderedFactor)
    y, X = unpack(df, ==(:Progression), rng=1234)
    if(balancing == "oversampling")
        println("Before oversamppling")
        Imbalance.checkbalance(y)
        oversampler = (@load RandomOversampler pkg=Imbalance verbosity=0)()
        mach = machine(oversampler)
        X, y = MLJ.transform(mach, X, y)
        println("After oversamppling")
    elseif(balancing == "SMOTE")
        println("Before SMOTE")
        Imbalance.checkbalance(y)
        oversampler = (@load SMOTE pkg=Imbalance verbosity=0)()
        mach = machine(oversampler)
        X, y = MLJ.transform(mach, X, y)
        println("After SMOTE")
    elseif(balancing == "no")
        println("No balancing")
    end
    Imbalance.checkbalance(y)
    (Xtrain, Xvalid), (ytrain, yvalid) = partition((X, y), 0.8, rng=123, multi=true, stratify=y)
    return Xtrain, ytrain, Xvalid, yvalid
end  # function DataPrepare

function TrainModelResearch(Xtrain, ytrain)

    disable_logging(Logging.Warn)
    best_models = []

    knn = (@load KNNClassifier verbosity=0)()
    K_range = range(knn, :K, lower=5, upper=20);
    self_tuning_knn = TunedModel(
        model=knn,
        resampling = StratifiedCV(nfolds=10, rng=1234),
        tuning = Grid(),
        range = K_range,
        measure = JaccardIndex(),
        operation = predict_mode
    )
    mach = machine(self_tuning_knn, Xtrain, ytrain)
    fit!(mach, verbosity=0)
    push!(best_models, mach)

    tree = (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)()
    m_depth_range = range(tree, :max_depth, lower=-1, upper=80);
    self_tuning_tree = TunedModel(
        model=tree,
        resampling = StratifiedCV(nfolds=10, rng=1234),
        tuning = Grid(),
        range = m_depth_range,
        measure = JaccardIndex(),
        operation = predict_mode
    )
    mach = machine(self_tuning_tree, Xtrain, ytrain)
    fit!(mach, verbosity=0)
    push!(best_models, mach)

    Forest = (@load RandomForestClassifier pkg=DecisionTree verbosity=0)()
    m_depth_range = range(Forest, :max_depth, lower=-1, upper=80)
    n_sub_range = range(Forest, :n_subfeatures, lower=-1, upper=18)
    n_tree_range = range(Forest, :n_trees, lower=2, upper=100)
    self_tuning_forest = TunedModel(
        model=Forest,
        resampling = StratifiedCV(nfolds=10, rng=1234),
        tuning = RandomSearch(),
        range = [m_depth_range, n_sub_range, n_tree_range],
        measure = JaccardIndex(),
        operation = predict_mode
    )
    mach = machine(self_tuning_forest, Xtrain, ytrain)
    fit!(mach, verbosity=0)
    push!(best_models, mach)

    SVC = (@load ProbabilisticSVC pkg=LIBSVM verbosity=0)()
    mach = machine(SVC, Xtrain, ytrain)
    fit!(mach, verbosity=0)
    push!(best_models, mach)

    LogClass = (@load LogisticClassifier pkg=MLJLinearModels verbosity=0)()
    mach = machine(LogClass, Xtrain, ytrain)
    fit!(mach, verbosity=0)
    push!(best_models, mach)

    return best_models
end

function bestModelsReport(best_models, Xvalid, yvalid)
    model_names = []
    bac = []
    jac = []
    f2 = []
    FN = []

    for model in best_models
        try
            push!(model_names, report(model).best_history_entry.model)
        catch
            push!(model_names, model.model)
        end
        ŷ = predict_mode(model, Xvalid)
        push!(bac, balanced_accuracy(ŷ, yvalid))
        push!(jac, JaccardIndex()(ŷ, yvalid))
        push!(f2, FScore(beta = 2)(ŷ, yvalid))
        push!(FN, falsenegative(ŷ, yvalid))
    end

    df_valid = DataFrame(Models=model_names, balanced_acc=bac, jaccard_index=jac, F2=f2, FN=FN)
    return df_valid
end  # function bestModelsReport

function ShapleyResearch(models, Xvalid, i, pl_tit)
    ϕ = shapley(Xvalid -> predict(models[i], Xvalid), Shapley.MonteCarlo(CPUThreads(), 1024), Xvalid)
    bar_data = []
    k = [string(i) for i in keys(ϕ)]
    for i in ϕ 
        push!(bar_data, mean(abs.(pdf.(i, 1))))
    end
    n = size(bar_data, 1)
    b = bar(
        bar_data,
        yticks=(1:1:n, k),
        ylims=(0, n+1),
        orientation=:horizontal,
        legend=false,
        xlims=(0, 0.3),
        title="Global feature importance",
        xlabel="Mean(abs(Shapley value))",
    )
    A = [pdf.(i, 1) for i in ϕ]
    v = violin(
        A, 
        xticks=(1:1:n, k),
        xlims=(0, n+1),
        legend=false,
        title="Local explanation summary",
        ylabel="SHAP value", 
        permute=(:y, :x),
    )
    plot(
        b,
        v,
        layout=(1, 2),
        plot_title=pl_tit,
        size=(1200, 900),
        margin=(20, :pt)
        )
end  # function ShapleyResearch

function metricTest()
    y = rand(Float64, 900)
    y = ifelse.(y .< 0.67, 1, 0)
    y = coerce(y, OrderedFactor)

    ŷ = rand(Float64, 900)
    # ŷ = y
    # ŷ = ifelse.(y .== 0 .&& ŷ .> 0.9, 0, 1)
    ŷ = ifelse.(y .== 1 .&& ŷ .> 0.9 .|| y .== 0 .&& ŷ .< 0.1, 1, 0)
    ŷ = coerce(ŷ, OrderedFactor)

    cm = ConfusionMatrix()(ŷ, y)
    @show fnr(cm)
    @show npv(cm)
    @show FScore(beta=2)(cm)
    @show FScore(beta=0.5)(cm)
    @show bac(cm)
    @show JaccardIndex()(cm)
    cm
end  # function metricTest 

end # module BrainMetastasis

using .BrainMetastasis