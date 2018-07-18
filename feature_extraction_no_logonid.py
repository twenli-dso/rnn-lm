import time
import json
import re
import os
import glob

file_list = glob.glob(os.path.join("../data/EventLogMonitorMarApr/",'*'))
#, '*'
#test_list = glob.glob(os.path.join(os.getcwd(),'TestData'))
#print(os.path.join(os.getcwd(),'TrainData', '*'))
#print(file_list)

traincounter = 0
testcounter = 0
allid = {}

def end_date_gen(start_date,numdays):
    date = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date = date + datetime.timedelta(days=numdays)
    return end_date.strftime('%Y%m%d')

##file_list = ['1461931342217-1_2016-04-29']

for url in file_list:
    with open(url) as data_file:    
        while True:
            try:
                data = data_file.readline().strip()
                if data == '':
                    break
                d = json.loads(data)
                eventtype = d['Log']['Event']['Name']
                if eventtype == 'Security':

                    traincounter += 1
                    
                    message = d['Log']['Event']['Message']
                    msg = re.split('[\r | \t]', message)
                    time = d['Log']['Event']['Timestamp-Readable-UTC']
                    
                    for i in range(len(msg)):
                        if msg[i] == '\n':
                            newmsg = ' '.join(msg[:i])
                            break
                        
                    procidno = 0
                    authpackno = 0
                    accountnameno = 0
                    domainno = 0
                    procnameno = 0
                    logonidno = 0
                    logontypeno = 0
                    eventidno = 0
                    templogonid = '?'
                    
                    timestamp = d['Log']['Event']['Timestamp-Readable-UTC']
                    logtime = timestamp[:10].replace('-','')
                    hostname = d['Log']['HostName']
                    eventid = d['Log']['Event']['EventID']
                    if d['Log']['Event']['EventID'] in allid:
                        allid[d['Log']['Event']['EventID']] += 1
                    else:
                        allid[d['Log']['Event']['EventID']] = 1
                            
                    for i in range (len(msg)):
                        if msg[i-1] == 'Account' and msg[i] == 'Name:' and accountnameno == 0:
                            accountname = msg[i+2]
                            accountnameno += 1
                        if msg[i-1] == 'Account' and msg[i] == 'Domain:' and domainno == 0:
                            domain = msg[i+2]
                            domainno += 1
                        if msg[i-1] == 'Process' and msg[i] == 'Name:' and procnameno == 0:
                            procname = msg[i+2]
                            procnameno += 1
                        if msg[i-1] == 'Process' and msg[i] == 'ID:' and procidno == 0:
                            procid = msg[i+2]
                            procidno += 1
                        if msg[i-1] == 'Logon' and msg[i] == 'Type:' and logontypeno == 0:
                            logontype = msg[i+3]
                            logontypeno += 1
                        if msg[i-1] == 'Authentication' and msg[i] == 'Package:' and authpackno == 0:
                            authpack = msg[i+1]
                            authpackno += 1
                        if msg[i-1] == 'Logon' and msg[i] == 'ID:':
                            templogonid = msg[i+2]
                            logonidno += 1

                    if procidno == 0:
                        procid = '?'
                    if authpackno == 0:
                        authpack = '?'
                    if procnameno == 0:
                        procname = '?'
                    logonid = templogonid
                    if accountnameno == 0:
                        accountname = '?'
                    if domainno == 0:
                        domain = '?'
                    if logontypeno == 0:
                        logontype = '?'

                    outfile = './output_2/data_security_' + str(logtime) + '.txt'

                    with open(outfile,'a') as out:
                        line = str(logtime) + ',' + str(hostname) + ',' + str(accountname) + ',' + str(domain) + ',' + str(eventid) + ',' + str(procname) + ',' + str(logontype) + ',' + str(authpack) + '\n'
                        out.write(line)

                if traincounter%200 == 1:
                    print('Current length', traincounter)
                        
            except KeyError:
                continue
            except UnicodeDecodeError:
                continue
            except TypeError:
                continue
            except json.decoder.JSONDecodeError:
                continue
print('Logs:',traincounter)
'''
for url in test_list:
    with open(url) as data_file:    
        while True:
            try:
                data = data_file.readline().strip()
                if data == '':
                    break
                d = json.loads(data)
                eventtype = d['Log']['Event']['Name']
                if eventtype == 'Security':

                    testcounter += 1
                    
                    message = d['Log']['Event']['Message']
                    msg = re.split('[\r | \t]', message)
                    time = d['Log']['Event']['Timestamp-Readable-UTC']
                    
                    for i in range(len(msg)):
                        if msg[i] == '\n':
                            newmsg = ' '.join(msg[:i])
                            break
                        
                    procidno = 0
                    authpackno = 0
                    accountnameno = 0
                    domainno = 0
                    procnameno = 0
                    logonidno = 0
                    logontypeno = 0
                    eventidno = 0
                    templogonid = '?'
                    
                    timestamp = d['Log']['Event']['Timestamp-Readable-UTC']
                    logtime = timestamp[:10].replace('-','')
                    hostname = d['Log']['HostName']
                    eventid = d['Log']['Event']['EventID']
                    if d['Log']['Event']['EventID'] in allid:
                        allid[d['Log']['Event']['EventID']] += 1
                    else:
                        allid[d['Log']['Event']['EventID']] = 1
                            
                    for i in range (len(msg)):
                        if msg[i-1] == 'Account' and msg[i] == 'Name:' and accountnameno == 0:
                            accountname = msg[i+2]
                            accountnameno += 1
                        if msg[i-1] == 'Account' and msg[i] == 'Domain:' and domainno == 0:
                            domain = msg[i+2]
                            domainno += 1
                        if msg[i-1] == 'Process' and msg[i] == 'Name:' and procnameno == 0:
                            procname = msg[i+2]
                            procnameno += 1
                        if msg[i-1] == 'Process' and msg[i] == 'ID:' and procidno == 0:
                            procid = msg[i+2]
                            procidno += 1
                        if msg[i-1] == 'Logon' and msg[i] == 'Type:' and logontypeno == 0:
                            logontype = msg[i+3]
                            logontypeno += 1
                        if msg[i-1] == 'Authentication' and msg[i] == 'Package:' and authpackno == 0:
                            authpack = msg[i+1]
                            authpackno += 1
                        if msg[i-1] == 'Logon' and msg[i] == 'ID:':
                            templogonid = msg[i+2]
                            logonidno += 1

                    if procidno == 0:
                        procid = '?'
                    if authpackno == 0:
                        authpack = '?'
                    if procnameno == 0:
                        procname = '?'
                    logonid = templogonid
                    if accountnameno == 0:
                        accountname = '?'
                    if domainno == 0:
                        domain = '?'
                    if logontypeno == 0:
                        logontype = '?'

                    outfile = 'data_' + str(logtime) + '.txt'

                    with open(outfile,'a') as out:
                        line = str(logtime) + ',' + str(hostname) + ',' + str(accountname) + ',' + str(domain) + ',' + str(logonid) + ',' + str(eventid) + ',' + str(procname) + ',' + str(logontype) + ',' + str(authpack) + '\n'
                        out.write(line)

                if testcounter%200 == 1:
                    print('Current length', testcounter)
                        
            except KeyError:
                continue
            except UnicodeDecodeError:
                continue
            except json.decoder.JSONDecodeError:
                continue

print('Testing Logs',testcounter)
'''
'''
for url in file_list:
    with open(url) as data_file:    
        while True:
            try:
                data = data_file.readline()
                d = json.loads(data)
                eventtype = d['Log']['Event']['Name']
                if eventtype == 'Security':

                    traincounter += 1
                    
                    message = d['Log']['Event']['Message']
                    msg = re.split('[\r | \t]', message)
                    time = d['Log']['Event']['Timestamp-Readable-UTC']
                    
                    for i in range(len(msg)):
                        if msg[i] == '\n':
                            newmsg = ' '.join(msg[:i])
                            break
                        
                    procidno = 0
                    authpackno = 0
                    accountnameno = 0
                    domainno = 0
                    procnameno = 0
                    logonidno = 0
                    logontypeno = 0
                    eventidno = 0
                    templogonid = '?'
                    
                    timestamp = d['Log']['Event']['Timestamp-Readable-UTC']
                    logtime = timestamp[:10].replace('-','')
                    timelst += [logtime,]
                    hostname += [d['Log']['HostName'],]
                    eventid += [d['Log']['Event']['EventID'],]
                    if d['Log']['Event']['EventID'] in allid:
                        allid[d['Log']['Event']['EventID']] += 1
                    else:
                        allid[d['Log']['Event']['EventID']] = 1
                            
                    for i in range (len(msg)):
                        if msg[i-1] == 'Account' and msg[i] == 'Name:' and accountnameno == 0:
                            accountname += [msg[i+2],]
                            accountnameno += 1
                        if msg[i-1] == 'Account' and msg[i] == 'Domain:' and domainno == 0:
                            domain += [msg[i+2],]
                            domainno += 1
                        if msg[i-1] == 'Process' and msg[i] == 'Name:' and procnameno == 0:
                            procname += [msg[i+2],]
                            procnameno += 1
                        if msg[i-1] == 'Process' and msg[i] == 'ID:' and procidno == 0:
                            procid += [msg[i+2],]
                            procidno += 1
                        if msg[i-1] == 'Logon' and msg[i] == 'Type:' and logontypeno == 0:
                            logontype += [msg[i+3],]
                            logontypeno += 1
                        if msg[i-1] == 'Authentication' and msg[i] == 'Package:' and authpackno == 0:
                            authpack += [msg[i+1],]
                            authpackno += 1
                        if msg[i-1] == 'Logon' and msg[i] == 'ID:':
                            templogonid = msg[i+2]
                            logonidno += 1

                    if procidno == 0:
                        procid += ['?',]
                    if authpackno == 0:
                        authpack += ['?',]
                    if procnameno == 0:
                        procname += ['?',]
                    logonid += [templogonid,]
                    print(templogonid)
                    if accountnameno == 0:
                        accountname += ['?',]
                    if domainno == 0:
                        domain += ['?',]
                    if logontypeno == 0:
                        logontype += ['?',]

                if len(procid)%200 == 1:
                    print('Current length', len(procid))
                        
            except KeyError:
                break
            except UnicodeDecodeError:
                break
            except json.decoder.JSONDecodeError:
                break

print('Training logs:',traincounter)

print(len(hostname), len(accountname), len(domain), len(procname), len(logontype), len(authpack),len(logonid), len(timelst))

# Format: Time, Host name, account name, domain, procid, logon type, auth pack, logon/off, success

outfile = 'train_data_sec3.txt'
'''
'''print('Writing train file\n\n')

with open(outfile,'a') as out:
    for i in range(len(hostname)):
        line = str(timelst[i]) + ',' + str(hostname[i]) + ',' + str(accountname[i]) + ',' + str(domain[i]) + ',' + str(logonid[i]) + ',' + str(eventid[i]) + ',' + str(procname[i]) + ',' + str(logontype[i]) + ',' + str(authpack[i]) + '\n'
        if i % 1000 == 0:
            print(line)
        out.write(line)
'''
'''
timelst = []
hostname = []
accountname = []
domain = []
procname = []
procid = []
logontype = []
authpack = []
on = []
success = []
logonid = []
eventid = []

for url in test_list:
    with open(url) as data_file:    
        while True:
            try:
                data = data_file.readline()
                d = json.loads(data)
                eventtype = d['Log']['Event']['Name']
                if eventtype == 'Security':

                    traincounter += 1
                    
                    message = d['Log']['Event']['Message']
                    msg = re.split('[\r | \t]', message)
                    time = d['Log']['Event']['Timestamp-Readable-UTC']
                    
                    for i in range(len(msg)):
                        if msg[i] == '\n':
                            newmsg = ' '.join(msg[:i])
                            break
                        
                    procidno = 0
                    authpackno = 0
                    accountnameno = 0
                    domainno = 0
                    procnameno = 0
                    logonidno = 0
                    logontypeno = 0
                    eventidno = 0
                    templogonid = '?'
                    
                    timestamp = d['Log']['Event']['Timestamp-Readable-UTC']
                    logtime = timestamp[:10].replace('-','')
                    timelst += [logtime,]
                    hostname += [d['Log']['HostName'],]
                    eventid += [d['Log']['Event']['EventID'],]
                    if d['Log']['Event']['EventID'] in allid:
                        allid[d['Log']['Event']['EventID']] += 1
                    else:
                        allid[d['Log']['Event']['EventID']] = 1
                            
                    for i in range (len(msg)):
                        if msg[i-1] == 'Account' and msg[i] == 'Name:' and accountnameno == 0:
                            accountname += [msg[i+2],]
                            accountnameno += 1
                        if msg[i-1] == 'Account' and msg[i] == 'Domain:' and domainno == 0:
                            domain += [msg[i+2],]
                            domainno += 1
                        if msg[i-1] == 'Process' and msg[i] == 'Name:' and procnameno == 0:
                            procname += [msg[i+2],]
                            procnameno += 1
                        if msg[i-1] == 'Process' and msg[i] == 'ID:' and procidno == 0:
                            procid += [msg[i+2],]
                            procidno += 1
                        if msg[i-1] == 'Logon' and msg[i] == 'Type:' and logontypeno == 0:
                            logontype += [msg[i+3],]
                            logontypeno += 1
                        if msg[i-1] == 'Authentication' and msg[i] == 'Package:' and authpackno == 0:
                            authpack += [msg[i+1],]
                            authpackno += 1
                        if msg[i-1] == 'Logon' and msg[i] == 'ID:':
                            templogonid = msg[i+2]
                            print(msg[i+2])
                            logonidno += 1

                    if procidno == 0:
                        procid += ['?',]
                    if authpackno == 0:
                        authpack += ['?',]
                    if procnameno == 0:
                        procname += ['?',]
                    logonid += [templogonid,]
                    print(templogonid)
                    if accountnameno == 0:
                        accountname += ['?',]
                    if domainno == 0:
                        domain += ['?',]
                    if logontypeno == 0:
                        logontype += ['?',]

                if len(procid)%200 == 1:
                    print('Current length', len(procid))
                        
            except KeyError:
                break
            except UnicodeDecodeError:
                break
            except json.decoder.JSONDecodeError:
                break

print('Testing logs:',testcounter)
'''
'''
print('Writing test file\n\n')

# Format: Time, Host name, account name, domain, procid, logon type, auth pack, logon/off, success

outfile2 = 'test_data_sec3.txt'

with open(outfile2,'a') as out:
    for i in range(len(hostname)):
        line = str(timelst[i]) + ',' + str(hostname[i]) + ',' + str(accountname[i]) + ',' + str(domain[i]) + ',' + str(logonid[i]) + ',' + str(eventid[i]) + ',' + str(procname[i]) + ',' + str(logontype[i]) + ',' + str(authpack[i]) + '\n'
        if i % 1000 == 0:
            print(line)
        out.write(line)

print(list(allid.keys()))
print(len(allid.keys()))

##print(time)
##print(hostname)
##print(accountname)
##print(domain)
##print(procname)
##print(procid)
##print(logontype)
##print(authpack)
'''
