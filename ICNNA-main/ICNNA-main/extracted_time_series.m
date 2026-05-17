icnna_startup;
%% 
% 请将 '文件名.mat' 替换为你实际的文件全名
load(['vc_data/icnna_CVC_PilotB_IntegrityChecked.mat'], 'E');
E.display();
%% 
% 确认受试者数量
nSubjects = E.nSubjects();
disp(['Total subjects found: ', num2str(nSubjects)]);
%% 
subj = E.getSubject(1); % 获取第一个受试者 [cite: 1049]

nSess = subj.nSessions(); % 获取该受试者的 Session 总数
disp(['受试者 1 的 Session 数量为: ', num2str(nSess)]);

subj2 = E.getSubject(2); % 获取第一个受试者 [cite: 1049]
nSess2 = subj2.nSessions(); % 获取该受试者的 Session 总数
disp(['受试者 2 的 Session 数量为: ', num2str(nSess2)]);
%% 
for i = 1:nSubjects
    subject = E.getSubject(i);
    
    % 动态获取该受试者的 session 数量（不会再因为写死数字而报错）
    nSess = subject.nSessions(); 
    
    for j = 1:nSess
        session = subject.getSession(j);
        datasource = session.getDataSource(1);
        structured_data = datasource.getStructuredData(datasource.activeStructured);
        
        % ---------------------------------------------------------
        % 提取 1: Time Series (时间序列数据)
        % ---------------------------------------------------------
        timestamps = array2table(structured_data.timeline.timestamps, 'VariableNames', {'timestamp'});
        
        data_sig_1 = array2table(structured_data.data(:,:,1));
        data_sig_1.signal(:) = 1;
        data_sig_1.subject(:) = i;
        data_sig_1.session(:) = j;
        data_sig_1 = horzcat(data_sig_1, timestamps);
        
        data_sig_2 = array2table(structured_data.data(:,:,2));
        data_sig_2.signal(:) = 2;
        data_sig_2.subject(:) = i;
        data_sig_2.session(:) = j;
        data_sig_2 = horzcat(data_sig_2, timestamps);
        
        sig_data_all = vertcat(data_sig_1, data_sig_2);
        
        % 导出时间序列 CSV
        ts_filename = "cvc_time_series_data/ts_subj_" + string(i) + "_sess_" + string(j) + ".csv";
        writetable(sig_data_all, ts_filename);
        
        % ---------------------------------------------------------
        % 提取 2: Event Information (事件标签数据)
        % ---------------------------------------------------------
        timeline = structured_data.timeline;
        
        % 【关键修改】：先用 struct2table 将结构体数组转换为表格
        cond_events = struct2table(timeline.condEvents);
        
        % 转换为表格后，就可以安全地批量添加新列了
        cond_events.subject(:) = i;
        cond_events.session(:) = j;
        
        % 导出事件 CSV
        event_filename = "cvc_event_data/event_subj_" + string(i) + "_sess_" + string(j) + ".csv";
        writetable(cond_events, event_filename);
    end
end

disp('=======================================');
disp('CVC 数据提取大功告成！快去文件夹里看看吧。');