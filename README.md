# FacebookLearner
Character prediction RNN thingo that learns to emulate a group chat hopefully maybe

## Input
The program expects `messages.json` to be a file that conforms to the following spec, note how to conform to `letters`, apostophes were replaced with whitespace:
```json
{
  "letters": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
  "authors":["alice", "bob"],
  "messages": [
    {
      "author": "alice",
      "message": "Hi bob here s my public key"
    },
    {
      "author": "bob",
      "message": "fuck you alice im sick of your shit"
    },
    {
      "author": "alice",
      "message": "REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
    }
  ]
}
```
