import random

def get_redis_host():
    p_subprocess = sp.Popen(["get_instance_by_service", "-ar", REDIS_BNS], stdout = sp.PIPE) 
    host_list = []
    while True:
        out = p_subprocess.stdout.readline()
        if out is None or out.strip() == '':
            break   
        s = out.strip().split(" ")
        if len(s) < 2:
            continue
        ip = s[1]
        start = out.find("proxy_port=")
        if start < 0:
            continue
        end = out.find(",stats")
        if end < 0:
            continue
        try:    
            port = int(out[start + len("proxy_port=") : end]) 
        except: 
            continue
        host_list.append([ip, port])
    if len(host_list) == 0:
        sys.stderr.write("failed to get redis host")
        exit(-1)
    random.shuffle(host_list)
    return host_list[0]
