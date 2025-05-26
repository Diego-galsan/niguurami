# Nigúurami
Nigúurami is a Tarahumara word that translates to "helper". True to its name, this AI agent is designed to support and assist in multi-agent interactions.
Developed on top of the OpenManus project, it features a simplified version of the A2A (Agent-to-Agent) communication protocol.


- You can test it using Postman
- Get :  
   - http://localhost:8000/.well-known/agent-card.json. 
   - http://localhost:8000/discovery/agents  
   - http://localhost:8000/discovery/agents/{{Agentid}}

- Post:  
   - http://localhost:8000/{{Agentid}}/message?Content-Type
       
   - Header:  
         Key: Content-Type  
         Values: application/json  
   - Body:  
        raw:  
                  {  
          "sender_agent_id": "postman-test-suite-001",  
          "message_type": "execute_capability",  
          "payload": {  
            "capability_name": "execute_prompt",  
            "params": {  
              "prompt": "Base on your knowladge base, what year México got the independency, just give me the indepence date and that's all."  
            }  
          }  
        }  
