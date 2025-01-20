Value = [0.78; 0.39; 11.41; 
         0.35; 0.73; 0.365; 
         50.14; 0.25; 9.81];

Units = ["m"; "m"; "kg"; 
         "kg*m^2"; "m"; "m"; 
         "kg"; "kg*m^2"; "m/s^2"];

Param = [ ...
    "Length, L1", "Center of mass, s1", "Mass, m1", ...
    "Moment of Inertia, J1", "Length, L2", ...
    "Center of mass, s2", "Mass, m2", ...
    "Moment of Inertia, J2", "Gravity, g"];
% CSVファイルの読み込み（ファイル名は適宜変更）
data = readtable('0.0873_3060_without_noise_delay.csv');

% データの抽出
q1 = data.q1;
q2 = data.q2;
dq1 = data.dq1;
dq2 = data.dq2;
u1 = data.u1;
u2 = data.u2;

% パラメータの設定（pvstate）
pvstate = [0.78; 0.39; 11.41; 0.35; 0.73; 0.365; 50.14; 0.25; 9.81];

% COPを格納する配列
num_steps = height(data);
COP_values = zeros(num_steps, 1);

% 各ステップでCOPを計算
for i = 1:num_steps
    % 状態ベクトルとトルクベクトルの作成
    x = [q1(i); q2(i); dq1(i); dq2(i)];
    tau = [u1(i); u2(i)];
    
    % COPの計算（makeCOP関数を使用）
    COP_values(i) = makeCOP(x, tau, pvstate);
end

% 結果の表示
figure;
plot(COP_values, 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('COP');
title('Center of Pressure (COP) Over Time');
grid on;

% 結果の保存（必要に応じて）
writematrix(COP_values, '0.0873COP_results.csv');
