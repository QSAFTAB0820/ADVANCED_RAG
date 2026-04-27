import sys #handle command line arguments
from agent import create_support_agent

def main():
    print("------------------------------------------------")
    print("Initializing Advanced Customer Support Bot...")
    try: 
        agent = create_support_agent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Please ensure your OPENAI_API_KEY is defined in a .env file.")
        sys.exit(1)

    print("Bot is ready! Type 'exit' or 'quit' to stop.")
    print("Example Queries:")
    print(" - What is your return policy?")
    print(" - I need help with my order 12345, and I also want to schedule a call.")
    print("------------------------------------------------")
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        print("Bot is thinking...")
        try:
          
            chat_history.append(("user", user_input))
            
            
            response = agent.invoke({"messages": chat_history})
            
            
            chat_history = response["messages"] # Updates the chat_history with the messages returned by the agent
            
          
            ai_message = chat_history[-1]
            outputMessage = ai_message.content
            print(f"\nBot: {outputMessage}")
            
            
            
        except Exception as e:
            print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
