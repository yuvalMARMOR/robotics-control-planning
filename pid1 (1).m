clc;

%% Define coordinates

timeStep = 0.1; % time step
timeVec = 1:timeStep:30;
numSamples = length(timeVec); % number of time samples

% Initial generalized coordinates 
angle1 = 0; 
angle2 = 0; 
angle3 = 0;
linearPos = 0;
angles = [angle1 angle2 angle3 linearPos];
 
% Define intermediate variables
cos1 = cosd(angle1);
sin1 = sind(angle1);
cos2 = cosd(angle2);
sin2 = sind(angle2);
cos3 = cosd(angle3);
sin3 = sind(angle3);

% Initial generalized velocities 
angle1Vel = 0; 
angle2Vel = 0; 
angle3Vel = 0;
linearVel = 0;
anglesVel = [angle1Vel angle2Vel angle3Vel linearVel];

angle1VelArr = zeros(1, numSamples);
angle2VelArr = zeros(1, numSamples);
angle3VelArr = zeros(1, numSamples);
linearVelArr = zeros(1, numSamples);

angle1Arr = zeros(1, numSamples);
angle2Arr = zeros(1, numSamples);
angle3Arr = zeros(1, numSamples);
linearArr = zeros(1, numSamples);

% Desired angles 
angle1_d = 90; 
angle2_d = 90; 
angle3_d = 90;
linear_d = 40;
angles_d = [angle1_d angle2_d angle3_d linear_d];

% Desired velocities 
angle1Vel_d = 0; 
angle2Vel_d = 0; 
angle3Vel_d = 0;
linearVel_d = 0;
anglesVel_d = [angle1Vel_d angle2Vel_d angle3Vel_d linearVel_d];

K = [1.5 1.5 1.5 1.5 30.595 14.2 19.68 29.85 0.005 0.005 0.005 0.01];

% Length of links
link1 = 187.5;
link2 = 200; 
link3 = 50; 
link4 = 200; 
link5 = 112.5; 
d4 = link5 + linearPos;

% Masses of links
mass1 = 350; 
mass2 = 341; 
mass3 = 100;
mass4 = 400;
mass5 = 600;
gravity = 9.81;

% Moments of inertia of links
inertia1 = 500;
inertia2 = 700; 
inertia3 = 400; 
inertia4 = 500;
inertia5 = 700;

integralError = [0 0 0 0];
totalError = [0 0 0 0];
prevError = [0 0 0 0];

integralErrorVel = [0 0 0 0];
totalErrorVel = [0 0 0 0];
prevErrorVel = [0 0 0 0];

%% Computing

for i = 1 : numSamples
    inertiaMat = [inertia2 0 0 0;
                  0 inertia3 0 0;
                  0 0 inertia4 0;
                  0 0 0 inertia5];

    inputTorques = [2 2 2 5]; % vector of moments for each motor joint
    d4 = link5 + linearPos;

    coriolisMat = [sin1 * (mass2 * gravity * link2 / 2 + mass3 * gravity * link3 + mass4 * gravity * link4 + mass5 * gravity * link5) - sin1 * cos2 * (mass3 * gravity * link3 / 2 + mass4 * gravity * link4 + mass5 * gravity * link5) - sin1 * cos2 * cos3 * (mass4 * gravity * link4 / 2 + mass5 * gravity * link5) + cos1 * sin2 * cos3 * (mass4 * gravity * link4 / 2 + mass5 * gravity * link5) + cos1 * sin2 * cos3 * mass5 * gravity * d4 / 2;
                  -cos1 * sin2 * (mass3 * gravity * link3 / 2 + mass4 * gravity * link4 + mass5 * gravity * link5) + sin1 * cos2 * (mass3 * gravity * link3 / 2 + mass4 * gravity * link4 + mass5 * gravity * link5) - cos1 * sin2 * cos3 * (mass4 * gravity * link4 / 2 + mass5 * gravity * link5) + sin1 * cos2 * cos3 * (mass4 * gravity * link4 / 2 + mass5 * gravity * link5) + sin1 * cos2 * cos3 * mass5 * gravity * d4 / 2;
                  -cos1 * cos2 * sin3 * (mass4 * gravity * link4 / 2 + mass5 * gravity * link5) - sin1 * sin2 * sin3 * (mass4 * gravity * link4 / 2 + mass5 * gravity * link5) - sin1 * sin2 * sin3 * mass5 * gravity * d4 / 2;
                  -mass5 * gravity / 2];

    errorAngles = angles_d - angles;
    integralError = integralError + errorAngles;
    errorVel = (errorAngles - prevError) / timeStep;

    prevError = errorAngles;
    
    % Compute torques
    torques = K(:,1:4).*errorAngles + K(:,5:8).*errorVel + K(:,9:12).*integralError;

    % Calculate angular accelerations
    angularAccel = inertiaMat \ (torques' * inputTorques) - inertiaMat \ (coriolisMat * anglesVel);

    % Update angular velocities
    angle1Vel = angularAccel(1) * timeStep + angle1Vel;
    angle2Vel = angularAccel(2) * timeStep + angle2Vel;
    angle3Vel = angularAccel(3) * timeStep + angle3Vel;
    linearVel = angularAccel(4) * timeStep + linearVel;

    angle1VelArr(i) = angle1Vel;
    angle2VelArr(i) = angle2Vel;
    angle3VelArr(i) = angle3Vel;
    linearVelArr(i) = linearVel;

    % Update angles
    angle1 = angularAccel(1) * timeStep * timeStep * 0.5 + angle1Vel * timeStep + angle1;
    angle2 = angularAccel(2) * timeStep * timeStep * 0.5 + angle2Vel * timeStep + angle2;
    angle3 = angularAccel(3) * timeStep * timeStep * 0.5 + angle3Vel * timeStep + angle3;
    linearPos = angularAccel(3) * timeStep * timeStep * 0.5 + linearVel * timeStep + linearPos;

    angle1Arr(i) = angle1;
    angle2Arr(i) = angle2;
    angle3Arr(i) = angle3;
    linearArr(i) = linearPos;

    % Define intermediate variables
    cos1_d = cosd(angle1);
    sin1_d = sind(angle1);
    cos2_d = cosd(angle2);
    sin2_d = sind(angle2);
    cos3_d = cosd(angle3);
    sin3_d = sind(angle3);

    cos1 = cosd(angle1);
    sin1 = sind(angle1);
    cos2 = cosd(angle2);
    sin2 = sind(angle2);
    cos3 = cosd(angle3);
    sin3 = sind(angle3);

    % Stick figure representation
    xCoords = [0, 0, -sin1*link2, -sin1*link2 + link3*(cos1*sin2 - cos2*sin1), -sin1*link2 + link3*(cos1*sin2 - cos2*sin1) + link4*(cos1*sin2*cos3 - sin1*cos2*cos3), -sin1*link2 + link3*(cos1*sin2 - cos2*sin1) + (link4 + d4)*(cos1*sin2*cos3 - sin1*cos2*cos3)];
    yCoords = [0, 0, 0, 0, sin3*link4, sin3*(link4 + d4)];
    zCoords = [0, link1, link1 + link2*cos1, link1 + link2*cos1 + link3*(cos1*cos2 + sin1*sin2), link1 + link2*cos1 + link3*(cos1*cos2 + sin1*sin2) + link4*(cos1*cos2*cos3 + sin1*sin2*cos3), link1 + link2*cos1 + link3*(cos1*cos2 + sin1*sin2) + (link4 + d4)*(cos1*cos2*cos3 + sin1*sin2*cos3)];

    plot3(xCoords, yCoords, zCoords, '-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
    grid on;
    
    if (angle1Arr(i) > angle1_d || angle2Arr(i) > angle2_d || angle3Arr(i) > angle3_d || linearArr(i) > linear_d)
        break; 
    end
    
    xlim([-300, 300]);
    ylim([-300, 300]);
    zlim([-100, 900]);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Stick Figure Representation of Robotic Arm');
end

%% Plotting

% Plot motion graphs
figure;
subplot(4, 1, 1);
plot(timeVec, angle1VelArr, 'r', 'LineWidth', 2);
xlabel('Time');
ylabel('\theta_1');
title('Joint Angle \theta_1_d_o_t vs Time');
grid on;

subplot(4, 1, 2);
plot(timeVec, angle2VelArr, 'g', 'LineWidth', 2);
xlabel('Time');
ylabel('\theta_2');
title('Joint Angle \theta_2_d_o_t vs Time');
grid on;

subplot(4, 1, 3);
plot(timeVec, angle3VelArr, 'b', 'LineWidth', 2);
xlabel('Time');
ylabel('\theta_3');
title('Joint Angle \theta_3_d_o_t vs Time');
grid on;

subplot(4, 1, 4);
plot(timeVec, linearVelArr, 'b', 'LineWidth', 2);
xlabel('Time');
ylabel('Linear');
title('Joint Linear Velocity vs Time');
grid on;

figure;
subplot(4, 1, 1);
plot(timeVec, angle1Arr, 'r', 'LineWidth', 2);
xlabel('Time');
ylabel('\theta_1');
title('Joint Angle \theta_1 vs Time');
grid on;

subplot(4, 1, 2);
plot(timeVec, angle2Arr, 'g', 'LineWidth', 2);
xlabel('Time');
ylabel('\theta_2');
title('Joint Angle \theta_2 vs Time');
grid on;

subplot(4, 1, 3);
plot(timeVec, angle3Arr, 'b', 'LineWidth', 2);
xlabel('Time');
ylabel('\theta_3');
title('Joint Angle \theta_3 vs Time');
grid on;

subplot(4, 1, 4);
plot(timeVec, linearArr, 'b', 'LineWidth', 2);
xlabel('Time');
ylabel('Linear');
title('Joint Linear Position vs Time');
grid on;

hold off;
