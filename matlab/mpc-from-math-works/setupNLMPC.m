function [msobj, coredata, onlinedata] = setupNLMPC(p, nx, nmv, Ts, pvstate, pvcost, x0, u0)
    % NL-MPCオブジェクトの作成
    msobj = nlmpcMultistage(p, nx, nmv);
    msobj.Ts = Ts;
    msobj.Model.StateFcn = "twolinkStateFcn";
    msobj.Model.StateJacFcn = "twolinkStateJacFcn";
    msobj.Model.ParameterLength = length(pvstate);

    % 各ステージのコスト関数の設定
    for k = 1:p+1
        msobj.Stages(k).CostFcn = "twolinkCostFcn";
        msobj.Stages(k).CostJacFcn = "twolinkCostJacFcn";
        msobj.Stages(k).ParameterLength = length(pvcost);
    end

    % 操作変数の制約設定
    msobj.ManipulatedVariables(1).Min = -30;
    msobj.ManipulatedVariables(1).Max =  30;
    msobj.ManipulatedVariables(2).Min = -60;
    msobj.ManipulatedVariables(2).Max =  60;

    % 状態変数の制約設定
    msobj.States(1).Min = -0.35;
    msobj.States(1).Max =  0.53;
    msobj.States(2).Min = -0.53;
    msobj.States(2).Max =  0.87;

    % シミュレーションデータの取得
    simdata = getSimulationData(msobj);
    simdata.StateFcnParameter = pvstate;
    simdata.StageParameter = repmat(pvcost, p+1, 1);

    % 最適化オプションの設定
    msobj.Optimization.Solver = "fmincon";
    msobj.Optimization.SolverOptions.MaxIterations = 5000;
    msobj.Optimization.SolverOptions.MaxFunctionEvaluations = 1e6;
    msobj.Optimization.SolverOptions.ConstraintTolerance = 1e-3;
    msobj.Optimization.SolverOptions.StepTolerance = 1e-8;
    msobj.Optimization.SolverOptions.FunctionTolerance = 1e-9;
    msobj.Optimization.SolverOptions.OptimalityTolerance = 1e-10;
    msobj.Optimization.SolverOptions.Display = "iter";
    msobj.Optimization.SolverOptions.ScaleProblem = true;
    msobj.Optimization.SolverOptions.Algorithm = 'sqp';

    % コード生成用データの取得とMEXファイル作成
    [coredata, onlinedata] = getCodeGenerationData(msobj, x0, u0, ...
        StateFcnParameter = pvstate, StageParameter = simdata.StageParameter);
    buildMEX(msobj, "nlmpcControllerMEX", coredata, onlinedata);
end
