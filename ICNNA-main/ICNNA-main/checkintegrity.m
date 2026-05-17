cd('C:\Users\Alice\Desktop\ic\individual project\Jingtong_fNIRS\Jingtong_fNIRS\csv_timeseries_iminroma\ICNNA-main\ICNNA-main\src')
icnna_startup;
load('cvc_data/icnna_ARSLA_CVC_IntegrityChecked.mat', 'E'); 

for i = 1:E.nSubjects()
    subject = E.getSubject(i);
    fprintf('\n===== Subject %d =====\n', i);

    for j = 1:subject.nSessions()
        session = subject.getSession(j);
        fprintf('Session %d\n', j);

        fprintf('  Number of data sources: %d\n', session.nDataSources());

        for ds = 1:session.nDataSources()
            datasource = session.getDataSource(ds);
            fprintf('  DataSource %d\n', ds);

            fprintf('    activeStructured = %d\n', datasource.activeStructured);
            fprintf('    number of structured data = %d\n', datasource.nStructuredData());

            for sd = 1:datasource.nStructuredData()
                structured_data = datasource.getStructuredData(sd);
                timestamps = structured_data.timeline.timestamps;

                fprintf('    StructuredData %d:\n', sd);
                fprintf('      timestamp min = %.2f\n', min(timestamps));
                fprintf('      timestamp max = %.2f\n', max(timestamps));
                fprintf('      n timestamps  = %d\n', length(timestamps));
                fprintf('      data size     = [%s]\n', num2str(size(structured_data.data)));
                
                timeline = structured_data.timeline;
                cond_events = timeline.condEvents;

                if ~isempty(cond_events)
                    disp('      condEvents preview:');
                    disp(head(cond_events));
                else
                    disp('      condEvents empty');
                end
            end
        end
    end
end