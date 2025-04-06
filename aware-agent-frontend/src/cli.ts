import { AwareAgent } from './app';
import { Command } from 'commander';
import { createInterface } from 'readline';

const program = new Command();

program
  .name('aware-agent')
  .description('CLI for interacting with the AwareAgent system')
  .version('0.1.0');

program
  .command('chat')
  .description('Start an interactive chat session with the agent')
  .option('-m, --model <model>', 'LLM model to use', 'mistral')
  .option('-p, --memory-path <path>', 'Path to memory storage', './memory')
  .option('-c, --context-path <path>', 'Path to context storage', './context')
  .action(async (options) => {
    try {
      const agent = new AwareAgent(options.model, options.memoryPath, options.contextPath);
      await agent.initialize();

      const rl = createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: '> '
      });

      console.log('AwareAgent CLI started. Type "exit" to quit, "export" to save session.\n');

      rl.prompt();

      rl.on('line', async (line) => {
        if (line.trim().toLowerCase() === 'exit') {
          rl.close();
          return;
        }

        if (line.trim().toLowerCase() === 'export') {
          const log = await agent.exportSession('markdown');
          console.log('\nSession Log:\n', log);
          rl.prompt();
          return;
        }

        try {
          const response = await agent.processMessage(line);
          console.log('\nAgent:', response);
        } catch (error) {
          console.error('Error:', error);
        }

        rl.prompt();
      });

      rl.on('close', () => {
        console.log('\nGoodbye!');
        process.exit(0);
      });
    } catch (error) {
      console.error('Failed to start agent:', error);
      process.exit(1);
    }
  });

program
  .command('export')
  .description('Export a session log')
  .option('-f, --format <format>', 'Export format (json or markdown)', 'json')
  .action(async (options) => {
    try {
      const agent = new AwareAgent();
      await agent.initialize();
      const log = await agent.exportSession(options.format);
      console.log(log);
    } catch (error) {
      console.error('Failed to export session:', error);
      process.exit(1);
    }
  });

program
  .command('status')
  .description('Get current agent status')
  .action(async () => {
    try {
      const agent = new AwareAgent();
      await agent.initialize();
      const state = agent.getState();
      console.log('Current Agent State:', JSON.stringify(state, null, 2));
    } catch (error) {
      console.error('Failed to get status:', error);
      process.exit(1);
    }
  });

program.parse(process.argv); 