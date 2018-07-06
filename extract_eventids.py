import json
from json.decoder import JSONDecodeError
from pathlib import Path

import glob, os
os.chdir("../data/EventLogMonitorMarApr/")

dict = {}
for filename in glob.glob("../data/EventLogMonitorMarApr/*"):
    print(filename)
    with open(filename, 'r') as infile:
        try:
            for line_num, line in enumerate(infile):
                line = infile.readline()
                jsonline = json.loads(line)
                event_name = jsonline['Log']['Event']['Name']
                event_id = jsonline['Log']['Event']['EventID']

                if event_name in dict:
                    existing_ids = dict[event_name]
                    if event_id not in existing_ids:
                        dict[event_name] = existing_ids + [event_id,]
                else: 
                    dict[event_name] = [event_id]
                
        except json.decoder.JSONDecodeError:
            continue
        except KeyError:
            continue
        except UnicodeDecodeError:
            continue
                    
with open ("./dictionary.json", 'w') as outfile:
    json.dump(dict, outfile, sort_keys=True, indent=4)
    
#get logs for unique security event ids
security_event_ids = dict['Security']
security_event_ids_logs = {}
for filename in glob.glob("*"):
    with open(filename, 'r') as infile:
        try:
            for line_num, line in enumerate(infile):
                line = infile.readline()
                jsonline = json.loads(line)

                event_id = jsonline['Log']['Event']['EventID']
                
                if event_id in security_event_ids:
                    #if event_id is not existing key in dict, add log line
                    if event_id not in security_event_ids_logs:
                        security_event_ids_logs[event_id] = jsonline
                
        except json.decoder.JSONDecodeError:
            continue
        except KeyError:
            continue
        except UnicodeDecodeError:
            continue

#print(security_event_ids_logs)
print(len(security_event_ids_logs))
with open ("security_event_logs.json", 'w') as outfile:
    json.dump(security_event_ids_logs, outfile, sort_keys=True, indent=4)

